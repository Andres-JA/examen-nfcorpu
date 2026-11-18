"""API del examen: python app.py (PORT configurable)."""
import os
from pathlib import Path
from flask import Flask, jsonify, request
from nfcorpus_ir import SearchEngine

def create_app(engine=None):
    app = Flask(__name__)
    engine = engine if engine is not None else SearchEngine(Path(__file__).parent / 'datos')

    @app.get('/health')
    def health():
        return jsonify(status='ok', corpus='nfcorpus', documents=len(engine.ids), models=['tfidf', 'bm25'])

    @app.post('/search')
    def search():
        if not request.is_json:
            return jsonify(error='Se requiere Content-Type: application/json'), 415
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error='El cuerpo debe ser un objeto JSON válido'), 400
        try:
            return jsonify(engine.search(payload.get('query'), payload.get('model', 'bm25'), payload.get('top_k', 10)))
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400

    return app

if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)
