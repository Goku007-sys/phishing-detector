# 🛡️ PhishGuard — ML Phishing URL Detector

PhishGuard is a machine learning web application that detects 
phishing URLs in real time. Built with Python, Flask, and 
scikit-learn, it analyzes 40+ structural features of any URL 
and classifies it as legitimate or phishing with 95% accuracy.

## 🔍 What it does
- Analyzes any URL instantly in under 15ms
- Extracts 40+ features like entropy, subdomains, TLD suspiciousness
- Uses an ensemble of Random Forest + Gradient Boosting
- Shows confidence score and human-readable risk breakdown
- Supports batch scanning of up to 20 URLs at once

## 🚀 Quick Start

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

## 🧠 ML Features Analyzed
- URL length, entropy, special characters
- Suspicious TLDs (.tk, .xyz, .ml, .cf)
- Brand keywords in subdomains
- IP address as hostname
- HTTPS presence
- URL shortener detection
- Phishing keyword count
- Subdomain depth analysis

## 🛠️ Tech Stack
- Backend: Python, Flask
- ML: scikit-learn, Random Forest, Gradient Boosting
- Frontend: HTML, CSS, JavaScript
- Libraries: NumPy, Pandas, BeautifulSoup4

## 🎯 Model Performance
- Test Accuracy: 95%
- Training data: 100 URLs (50 legitimate + 50 phishing)
- Features extracted: 40+
- Analysis time: ~15ms per URL

## 👨‍💻 Made by
Goku007-sys