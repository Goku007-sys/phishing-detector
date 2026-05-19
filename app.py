"""
Phishing Detector - Flask Web Application
"""
import json
import time
from flask import Flask, render_template, request, jsonify
from ml_model import predict_url, get_model, train_model

app = Flask(__name__)

# Pre-train model on startup
print("[App] Initializing ML model...")
get_model()
print("[App] Model ready.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = (data or {}).get('url', '').strip()

    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if len(url) > 2048:
        return jsonify({'error': 'URL too long (max 2048 chars)'}), 400

    t0 = time.time()
    try:
        result = predict_url(url)
        result['analysis_time_ms'] = round((time.time() - t0) * 1000, 1)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch', methods=['POST'])
def batch_predict():
    data = request.get_json()
    urls = (data or {}).get('urls', [])
    if not urls or not isinstance(urls, list):
        return jsonify({'error': 'Provide a list of URLs'}), 400
    if len(urls) > 20:
        return jsonify({'error': 'Max 20 URLs per batch'}), 400

    results = []
    for url in urls:
        try:
            results.append(predict_url(url.strip()))
        except Exception as e:
            results.append({'url': url, 'error': str(e)})
    return jsonify({'results': results, 'count': len(results)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
