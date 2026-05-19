"""
Phishing URL Detection - ML Model
Features extracted from URL structure, domain info, and content signals.
"""

import re
import math
import urllib.parse
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

SUSPICIOUS_TLDS = {'.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.pw', '.top',
                   '.click', '.link', '.club', '.online', '.site', '.info',
                   '.loan', '.win', '.bid', '.stream', '.download'}

BRAND_KEYWORDS = [
    'paypal', 'apple', 'amazon', 'google', 'microsoft', 'facebook', 'instagram',
    'twitter', 'netflix', 'bank', 'secure', 'account', 'login', 'signin',
    'verify', 'update', 'confirm', 'password', 'ebay', 'chase', 'wellsfargo',
    'citibank', 'hsbc', 'barclays', 'dropbox', 'linkedin', 'whatsapp', 'telegram'
]

PHISH_WORDS = [
    'login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm',
    'banking', 'password', 'credential', 'suspend', 'limited', 'unusual',
    'alert', 'urgent', 'free', 'winner', 'prize', 'lucky', 'claim'
]


def shannon_entropy(s):
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def extract_features(url: str) -> dict:
    """Extract 30+ features from a URL for ML classification."""
    features = {}

    # Normalize
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        parsed = urllib.parse.urlparse('http://unknown.com')

    full_url = url.lower()
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    query = parsed.query or ''

    # ── URL-level features ──
    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)
    features['query_length'] = len(query)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_question_marks'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_ampersands'] = url.count('&')
    features['num_at_signs'] = url.count('@')
    features['num_percent'] = url.count('%')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['digit_ratio'] = features['num_digits'] / max(len(url), 1)
    features['num_params'] = len(urllib.parse.parse_qs(query))

    # ── Security indicators ──
    features['has_https'] = int(parsed.scheme == 'https')
    features['has_ip_address'] = int(bool(re.match(
        r'^(\d{1,3}\.){3}\d{1,3}$', hostname)))
    features['has_port'] = int(parsed.port is not None and parsed.port not in (80, 443))
    features['has_double_slash'] = int('//' in path)
    features['has_at_symbol'] = int('@' in url)
    features['has_hex_encoding'] = int('%' in url)
    features['has_data_uri'] = int(url.startswith('data:'))

    # ── Subdomain analysis ──
    parts = hostname.split('.')
    features['num_subdomains'] = max(len(parts) - 2, 0)
    features['subdomain_length'] = len('.'.join(parts[:-2])) if len(parts) > 2 else 0
    features['has_www'] = int(hostname.startswith('www.'))

    # ── TLD suspiciousness ──
    tld = '.' + parts[-1] if parts else ''
    features['suspicious_tld'] = int(tld in SUSPICIOUS_TLDS)
    features['tld_length'] = len(parts[-1]) if parts else 0

    # ── Brand / keyword signals ──
    features['brand_in_subdomain'] = int(any(
        brand in '.'.join(parts[:-2]).lower() for brand in BRAND_KEYWORDS))
    features['brand_in_path'] = int(any(brand in path.lower() for brand in BRAND_KEYWORDS))
    features['phish_word_count'] = sum(w in full_url for w in PHISH_WORDS)
    features['brand_keyword_count'] = sum(b in full_url for b in BRAND_KEYWORDS)

    # ── Entropy ──
    features['url_entropy'] = shannon_entropy(url)
    features['hostname_entropy'] = shannon_entropy(hostname)
    features['path_entropy'] = shannon_entropy(path)

    # ── Structural oddities ──
    features['has_redirect'] = int('redirect' in full_url or 'redir' in full_url or
                                   'forward' in full_url or 'goto' in full_url)
    features['has_shortener'] = int(any(s in hostname for s in [
        'bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'buff.ly',
        'short', 'tiny', 'rebrand.ly', 'cutt.ly']))
    features['consecutive_digits'] = len(re.findall(r'\d{4,}', url))
    features['longest_word'] = max((len(w) for w in re.split(r'[./\-_?=&]', url) if w), default=0)
    features['num_special_chars'] = sum(not c.isalnum() for c in url)

    return features


def features_to_vector(features: dict) -> np.ndarray:
    FEATURE_ORDER = [
        'url_length', 'hostname_length', 'path_length', 'query_length',
        'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
        'num_question_marks', 'num_equals', 'num_ampersands', 'num_at_signs',
        'num_percent', 'num_digits', 'digit_ratio', 'num_params',
        'has_https', 'has_ip_address', 'has_port', 'has_double_slash',
        'has_at_symbol', 'has_hex_encoding', 'has_data_uri',
        'num_subdomains', 'subdomain_length', 'has_www',
        'suspicious_tld', 'tld_length',
        'brand_in_subdomain', 'brand_in_path',
        'phish_word_count', 'brand_keyword_count',
        'url_entropy', 'hostname_entropy', 'path_entropy',
        'has_redirect', 'has_shortener', 'consecutive_digits',
        'longest_word', 'num_special_chars'
    ]
    return np.array([features.get(k, 0) for k in FEATURE_ORDER])


# ─────────────────────────────────────────────
# Training Data (balanced synthetic + patterns)
# ─────────────────────────────────────────────

LEGIT_URLS = [
    "https://www.google.com/search?q=python",
    "https://www.amazon.com/dp/B08N5KWB9H",
    "https://github.com/openai/gpt-4",
    "https://stackoverflow.com/questions/12345",
    "https://www.wikipedia.org/wiki/Machine_learning",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.microsoft.com/en-us/windows",
    "https://docs.python.org/3/library/os.html",
    "https://www.bbc.com/news/world",
    "https://www.nytimes.com/2024/01/01/technology",
    "https://www.linkedin.com/in/johndoe",
    "https://www.reddit.com/r/MachineLearning",
    "https://www.apple.com/iphone",
    "https://www.netflix.com/title/12345",
    "https://www.paypal.com/us/home",
    "https://www.dropbox.com/home",
    "https://mail.google.com/mail/u/0/",
    "https://drive.google.com/drive/my-drive",
    "https://www.twitter.com/home",
    "https://www.instagram.com/explore",
    "https://www.facebook.com/",
    "https://www.ebay.com/sch/i.html?_nkw=laptop",
    "https://www.chase.com/personal/bank-accounts",
    "https://www.wellsfargo.com/",
    "https://www.coursera.org/learn/machine-learning",
    "https://www.udemy.com/course/python-bootcamp",
    "https://www.medium.com/tag/python",
    "https://www.cloudflare.com/",
    "https://www.adobe.com/products/photoshop.html",
    "https://www.zoom.us/meeting",
    "https://www.slack.com/intl/en-in/",
    "https://www.spotify.com/us/",
    "https://www.airbnb.com/",
    "https://www.booking.com/",
    "https://www.tripadvisor.com/",
    "https://store.steampowered.com/",
    "https://www.twitch.tv/",
    "https://www.heroku.com/",
    "https://aws.amazon.com/",
    "https://azure.microsoft.com/",
    "https://console.cloud.google.com/",
    "https://www.mongodb.com/cloud/atlas",
    "https://www.postgresql.org/",
    "https://www.djangoproject.com/",
    "https://flask.palletsprojects.com/",
    "https://www.tensorflow.org/",
    "https://pytorch.org/",
    "https://scikit-learn.org/stable/",
    "https://pandas.pydata.org/",
    "https://numpy.org/",
]

PHISH_URLS = [
    "http://paypal-security-update.tk/login?redirect=https://paypal.com",
    "http://192.168.1.1/secure-login/paypal/verify",
    "https://apple-id-verify.xyz/signin?user=1234",
    "http://amazon-order-confirm.ml/account/login",
    "http://secure-banking.gq/chase/login.php",
    "http://microsoft-alert.cf/update-account?id=98765",
    "http://google-verification.pw/gmail/verify?token=abc123",
    "http://netflix-billing.tk/update-payment",
    "http://ebay-security.ga/signin?redirect=1",
    "http://facebook-login-verify.xyz/account",
    "http://wellsfargo-alert.ml/secure/login",
    "http://dropbox-share.cf/view?file=invoice.pdf",
    "http://linkedin-profile.gq/login/verify",
    "http://instagram-verify.tk/confirm?code=xxx",
    "http://twitter-security.pw/signin",
    "http://apple-support.ml/id/refund",
    "http://paypa1.com/login",
    "http://amaz0n.com/account/login",
    "http://micosoft.com/windowsupdate",
    "http://faceb00k.com/login.php",
    "https://secure-paypal-login.ru/signin",
    "http://account-verify.top/paypal/secure",
    "http://update-your-account.click/banking",
    "http://free-iphone-winner.online/claim",
    "http://bit.ly/3xFakeLink",
    "http://tinyurl.com/suspicious-link",
    "http://paypal.com.phish-site.xyz/login",
    "http://amazon.com.verify-account.tk/",
    "http://apple.com.id-verify.ml/signin",
    "http://chase.com.secure-login.gq/",
    "http://login-paypal-secure.pw/verify?user=victim",
    "http://secure.paypal.com.login.phishingsite.com/",
    "http://www.paypal.com.account-suspended.ru/login",
    "http://signin.amazon.co.uk.phish.xyz/",
    "http://www.google.com.update-required.tk/",
    "http://account.microsoft.com.verify.ml/",
    "http://id.apple.com.unlock.cf/",
    "http://support.netflix.com.update.gq/payment",
    "http://www.facebook.com.alert-security.pw/",
    "https://193.142.58.1/banking/login",
    "https://10.0.0.1/secure/paypal",
    "http://login.verify.update.secure.banking.com.tk/",
    "http://xyz-free-winner-prize-claim.top/",
    "http://urgent-verify-account-suspended-alert.xyz/login",
    "http://confirm-your-identity-now.ml/secure",
    "http://update-payment-method-required.cf/netflix",
    "http://your-account-has-been-limited.gq/paypal",
    "http://unusual-activity-detected-verify.pw/bank",
    "http://click-here-to-claim-prize.top/winner",
    "http://download-free-software.xyz/install.exe",
]


def build_training_data():
    X, y = [], []
    for url in LEGIT_URLS:
        f = extract_features(url)
        X.append(features_to_vector(f))
        y.append(0)
    for url in PHISH_URLS:
        f = extract_features(url)
        X.append(features_to_vector(f))
        y.append(1)
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def build_model():
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                min_samples_split=2, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                     max_depth=5, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)],
                                voting='soft', weights=[2, 1])
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', ensemble)])
    return pipeline


_model = None

def get_model():
    global _model
    if _model is None:
        _model = train_model()
    return _model


def train_model():
    print("[ML] Building training dataset...")
    X, y = build_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model()
    print("[ML] Training ensemble model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[ML] Test accuracy: {acc:.2%}")
    return model


def predict_url(url: str):
    """Return (label, confidence, features, risk_breakdown)."""
    model = get_model()
    features = extract_features(url)
    vec = features_to_vector(features).reshape(1, -1)

    proba = model.predict_proba(vec)[0]
    phish_prob = float(proba[1])
    label = 'phishing' if phish_prob >= 0.5 else 'legitimate'

    # Risk breakdown
    risk_factors = []
    if features['has_ip_address']:
        risk_factors.append({'factor': 'IP address as hostname', 'severity': 'high'})
    if features['suspicious_tld']:
        risk_factors.append({'factor': 'Suspicious TLD', 'severity': 'high'})
    if features['brand_in_subdomain']:
        risk_factors.append({'factor': 'Brand name in subdomain', 'severity': 'high'})
    if features['phish_word_count'] > 2:
        risk_factors.append({'factor': f'{int(features["phish_word_count"])} phishing keywords found', 'severity': 'medium'})
    if not features['has_https']:
        risk_factors.append({'factor': 'No HTTPS', 'severity': 'medium'})
    if features['has_at_symbol']:
        risk_factors.append({'factor': '@ symbol in URL (redirection trick)', 'severity': 'high'})
    if features['num_subdomains'] > 3:
        risk_factors.append({'factor': f'Excessive subdomains ({int(features["num_subdomains"])})', 'severity': 'medium'})
    if features['url_length'] > 100:
        risk_factors.append({'factor': 'Very long URL', 'severity': 'low'})
    if features['has_redirect']:
        risk_factors.append({'factor': 'Redirect keyword in URL', 'severity': 'medium'})
    if features['has_shortener']:
        risk_factors.append({'factor': 'URL shortener detected', 'severity': 'medium'})
    if features['hostname_entropy'] > 4:
        risk_factors.append({'factor': 'High hostname entropy (randomized)', 'severity': 'medium'})
    if features['num_hyphens'] > 4:
        risk_factors.append({'factor': f'Many hyphens ({int(features["num_hyphens"])})', 'severity': 'low'})

    return {
        'url': url,
        'label': label,
        'confidence': round(phish_prob * 100, 1),
        'safe_score': round((1 - phish_prob) * 100, 1),
        'features': {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in features.items()},
        'risk_factors': risk_factors
    }


if __name__ == '__main__':
    # Quick self-test
    test_urls = [
        "https://www.google.com",
        "http://paypal-verify.tk/login?redirect=paypal.com",
        "http://192.168.0.1/banking/secure",
        "https://www.amazon.com/dp/B08N5KWB9H",
        "http://apple-id-verify.xyz/signin",
    ]
    for u in test_urls:
        r = predict_url(u)
        print(f"[{r['label'].upper():12}] {r['confidence']:5.1f}%  {u}")
