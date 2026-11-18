"""Recuperación de documentos completos de BeIR/NFCorpus."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

STEMMER = SnowballStemmer('english')

@lru_cache(maxsize=100000)
def stem(token):
    return STEMMER.stem(token)

def preprocess(text):
    normalized = unicodedata.normalize('NFKC', str(text)).lower()
    tokens = re.findall(r'[a-z0-9]+', normalized)
    return [stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS]

class SearchEngine:
    def __init__(self, data_dir='datos', k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        paths = sorted((Path(data_dir) / 'nfcorpus' / 'corpus').glob('*.parquet'))
        if not paths:
            raise FileNotFoundError('Primero descarga datos/nfcorpus desde Hugging Face.')
        self.documents = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        self.documents['_id'] = self.documents['_id'].astype(str)
        self.documents = self.documents.sort_values('_id').reset_index(drop=True)
        assert self.documents['_id'].is_unique, 'Identificadores duplicados'
        self.ids = self.documents['_id'].tolist()
        texts = (self.documents['title'].fillna('') + ' ' + self.documents['text'].fillna('')).tolist()
        self.tokens = [preprocess(t) for t in texts]
        self.lengths = np.array([len(t) for t in self.tokens], dtype=float)
        self.avgdl = float(self.lengths.mean())
        self.inverted_index = defaultdict(list)
        for row, tokens in enumerate(self.tokens):
            for term, frequency in Counter(tokens).items():
                self.inverted_index[term].append((row, frequency))
        self.vocabulary = sorted(self.inverted_index)
        self.vectorizer = TfidfVectorizer(
            analyzer=preprocess, lowercase=False, norm='l2',
            smooth_idf=True, sublinear_tf=True, dtype=np.float64,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.bm25_idf = {
            term: math.log(1 + (len(self.ids) - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.inverted_index.items()
        }

    def search(self, query, model='bm25', top_k=10):
        if model not in ('bm25', 'tfidf'):
            raise ValueError('model debe ser bm25 o tfidf')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query debe ser un texto no vacío')
        if type(top_k) is not int or not 1 <= top_k <= len(self.ids):
            raise ValueError(f'top_k debe ser un entero entre 1 y {len(self.ids)}')
        scores = np.zeros(len(self.ids), dtype=float)
        if model == 'tfidf':
            q = self.vectorizer.transform([query])
            scores = (self.tfidf_matrix @ q.T).toarray().ravel()
        else:
            # Cada término distinto de la consulta participa una vez (BM25 sin k3).
            for term in set(preprocess(query)):
                postings = self.inverted_index.get(term)
                if not postings:
                    continue
                rows, frequencies = zip(*postings)
                rows = np.array(rows, dtype=int)
                tf = np.array(frequencies, dtype=float)
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[rows] / self.avgdl)
                scores[rows] += self.bm25_idf[term] * tf * (self.k1 + 1) / denominator
        # Desempate determinista por ID; también completa top_k con puntuaciones 0.
        order = np.argsort(-scores, kind='stable')[:top_k]
        results = []
        for rank, row in enumerate(order, 1):
            doc = self.documents.iloc[int(row)]
            results.append(dict(rank=rank, document_id=doc['_id'], score=float(scores[row]),
                                title=str(doc['title']), snippet=str(doc['text'])[:250]))
        return dict(model=model, results=results)

def load_test(data_dir='datos'):
    base = Path(data_dir)
    paths = sorted((base / 'nfcorpus' / 'queries').glob('*.parquet'))
    queries = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    queries['_id'] = queries['_id'].astype(str)
    assert queries['_id'].is_unique
    qrels_frame = pd.read_csv(base / 'nfcorpus-qrels' / 'test.tsv', sep='\t',
                              dtype={'query-id': str, 'corpus-id': str})
    qrels = defaultdict(dict)
    for qid, docid, score in qrels_frame[['query-id', 'corpus-id', 'score']].itertuples(index=False, name=None):
        qrels[qid][docid] = max(float(score), qrels[qid].get(docid, 0))
    test_queries = queries[queries['_id'].isin(qrels)].sort_values('_id').reset_index(drop=True)
    assert set(test_queries['_id']) == set(qrels), 'Faltan consultas test'
    return test_queries, dict(qrels), qrels_frame

def metrics_at_10(document_ids, judgments):
    relevant = {docid for docid, value in judgments.items() if value > 0}
    hits = sum(docid in relevant for docid in document_ids[:10])
    # Ganancia lineal, compatible con la convención de trec_eval/BEIR.
    dcg = sum(judgments.get(docid, 0) / math.log2(rank + 2)
              for rank, docid in enumerate(document_ids[:10]))
    ideal = sorted((s for s in judgments.values() if s > 0), reverse=True)[:10]
    idcg = sum(s / math.log2(rank + 2) for rank, s in enumerate(ideal))
    return {'Precision@10': hits / 10,
            'Recall@10': hits / len(relevant) if relevant else 0.0,
            'NDCG@10': dcg / idcg if idcg else 0.0}

def evaluate(engine, queries, qrels):
    records, rankings = [], {'tfidf': {}, 'bm25': {}}
    assert all(docid in set(engine.ids) for js in qrels.values() for docid in js)
    for model in rankings:
        for qid, query in queries[['_id', 'text']].itertuples(index=False, name=None):
            response = engine.search(query, model, 10)
            rankings[model][qid] = response['results']
            ids = [r['document_id'] for r in response['results']]
            records.append({'model': model, 'query_id': qid, **metrics_at_10(ids, qrels[qid])})
    details = pd.DataFrame(records)
    summary = details.groupby('model')[['Precision@10', 'Recall@10', 'NDCG@10']].mean()
    return details, summary, rankings
