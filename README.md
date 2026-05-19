# 🛡️ PhishGuard — ML Phishing URL Detector

A fully functional machine learning web application that detects phishing URLs in real time using an ensemble of Random Forest + Gradient Boosting classifiers trained on 40+ URL structural features.

---

## Features

- **40+ extracted features** per URL: entropy, subdomain depth, suspicious TLDs, brand keywords, IP detection, etc.
- **Ensemble ML model**: Voting classifier combining Random Forest (200 trees) + Gradient Boosting
- **Risk factor breakdown**: Human-readable explanation of each threat signal
- **Batch scanning**: Analyze up to 20 URLs at once via REST API
- **Dark-themed, responsive UI** with real-time confidence gauges
- **REST API**: Integrate into any system via `/api/predict` and `/api/batch`

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
Visit: http://localhost:5000

---

## REST API

### Single URL
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-verify.tk/login"}'
```

**Response:**
```json
{
  "url": "http://paypal-verify.tk/login",
  "label": "phishing",
  "confidence": 94.3,
  "safe_score": 5.7,
  "risk_factors": [
    {"factor": "Suspicious TLD", "severity": "high"},
    {"factor": "Brand name in subdomain", "severity": "high"}
  ],
  "features": { "url_length": 35, "has_https": 0, ... },
  "analysis_time_ms": 12.4
}
```

### Batch (up to 20 URLs)
```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://google.com", "http://phish.tk/login"]}'
```

---

## ML Features Extracted

| Category | Features |
|----------|----------|
| URL Length | url_length, hostname_length, path_length, query_length |
| Special Chars | num_dots, hyphens, slashes, @, %, =, & |
| Security | has_https, has_ip_address, has_port, has_at_symbol |
| Subdomains | num_subdomains, subdomain_length, has_www |
| TLD | suspicious_tld, tld_length |
| Content | brand_in_subdomain, phish_word_count, has_redirect |
| Entropy | url_entropy, hostname_entropy, path_entropy |
| Structure | has_shortener, consecutive_digits, longest_word |

---

## Project Structure

```
phishing_detector/
├── app.py          # Flask web application
├── ml_model.py     # Feature extraction + ML model
├── requirements.txt
├── README.md
└── templates/
    └── index.html  # Web UI
```

---

## Extending the Model

To improve accuracy with real-world data:
1. Download the [UCI Phishing Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites) or [PhishTank](https://www.phishtank.com/developer_info.php)
2. Extract features using `extract_features()` from `ml_model.py`
3. Re-train using `build_model().fit(X, y)`

---

> ⚠️ This tool is for educational and research purposes. Not a substitute for professional security tools.
