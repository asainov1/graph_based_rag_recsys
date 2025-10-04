#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading News Parser (Hybrid BS4 + Docling optional)

Extracts structured info from finance/trading news HTML:
- title, author, date
- cleaned body paragraphs
- key sentences
- tickers detected in multiple formats: $TSLA, (NASDAQ:AAPL), NYSE:IBM, AAPL
- prices and % changes
- naive finance sentiment score
- export utilities (JSONL/CSV)

Author: ChatGPT
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from bs4 import BeautifulSoup

try:
    from docling.document_converter import DocumentConverter
    _HAS_DOCLING = True
except Exception:
    _HAS_DOCLING = False

# -------------------------
# Data model
# -------------------------
@dataclass
class NewsRecord:
    url: Optional[str]
    title: str
    author: Optional[str]
    published: Optional[str]
    tickers: List[str]
    prices: List[str]
    percents: List[str]
    key_sentences: List[str]
    sentiment: float
    body: str
    source: Optional[str] = None

# -------------------------
# Finance lexicons
# -------------------------
FIN_BULLISH = {
    "beat","beats","surge","surged","rally","rallied","soar","soared","gain","gains","gained",
    "upgrade","upgrades","raised","raises","increase","increased","record","breakout","bullish",
    "strong","growth","optimistic","outperform","overweight","buy","accumulate","profit","profits",
}
FIN_BEARISH = {
    "miss","misses","plunge","plunged","drop","dropped","fell","fall","cut","cuts","cutting",
    "downgrade","downgrades","lowered","lower","decline","declined","weak","bearish","loss","losses",
    "warning","halt","bankrupt","bankruptcy","investigation","lawsuit","guidance cut","missed",
}

# -------------------------
# Regex patterns
# -------------------------
TICKER_RE = re.compile(
    r'\b(?:\$[A-Z]{1,5}|\((?:NASDAQ|NYSE|AMEX|LSE|SIX|TSX):[A-Z.]{1,6}\)|'
    r'(?:NASDAQ|NYSE|AMEX|LSE|SIX|TSX):[A-Z.]{1,6}|[A-Z]{1,5})(?=\b)'
)
PERCENT_RE = re.compile(r'[-+]?\d+(?:\.\d+)?\s*%')
PRICE_RE = re.compile(r'\$\s?\d{1,4}(?:,\d{3})*(?:\.\d+)?')
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

# -------------------------
# Helpers
# -------------------------
def _clean_text(txt: str) -> str:
    return re.sub(r'\s+', ' ', txt).strip()

def _extract_meta(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    title = None
    if soup.title and soup.title.text:
        title = _clean_text(soup.title.text)
    ogt = soup.find('meta', property='og:title')
    if ogt and ogt.get('content'):
        title = _clean_text(ogt['content'])

    author = None
    for sel in [('meta', {'name':'author'}),
                ('meta', {'property':'article:author'}),
                ('span', {'class': re.compile('author', re.I)}),
                ('a', {'rel':'author'})]:
        tag = soup.find(*sel) if isinstance(sel, tuple) else soup.find(sel)
        if tag:
            author = _clean_text(tag.get('content') or tag.get_text())
            break

    published = None
    for prop in ['article:published_time','og:updated_time','pubdate','date']:
        tag = soup.find('meta', {'property': prop}) or soup.find('meta', {'name': prop})
        if tag and tag.get('content'):
            published = tag['content']
            break
    time_tag = soup.find('time')
    if not published and time_tag:
        published = _clean_text(time_tag.get('datetime') or time_tag.get_text())

    site_name = None
    msn = soup.find('meta', property='og:site_name')
    if msn and msn.get('content'):
        site_name = msn['content']

    return {'title': title, 'author': author, 'published': published, 'source': site_name}

def _extract_body_bs4(soup: BeautifulSoup) -> List[str]:
    for t in soup(['script','style','noscript','header','footer','aside','form','svg']):
        t.decompose()

    candidates = []
    selectors = [
        ('article', {}),
        ('div', {'itemprop':'articleBody'}),
        ('div', {'class': re.compile(r'(article|content|post)-?body', re.I)}),
        ('section', {'class': re.compile(r'(article|content)', re.I)}),
    ]
    for name, attrs in selectors:
        candidates.extend(soup.find_all(name, attrs=attrs))
    if not candidates:
        candidates = [soup.body or soup]

    paras = []
    for node in candidates:
        for fig in node.find_all('figure'):
            cap = fig.find('figcaption')
            if cap:
                txt = _clean_text(cap.get_text(' '))
                if len(txt) > 30:
                    paras.append(txt)
            fig.decompose()
        for p in node.find_all(['p','li']):
            txt = _clean_text(p.get_text(' '))
            if len(txt) >= 30 and not txt.lower().startswith(('advertisement','ad ')):
                paras.append(txt)

    seen, out = set(), []
    for p in paras:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _extract_tickers(text: str) -> List[str]:
    matches = set()
    for m in TICKER_RE.findall(text):
        raw = m.strip('()').replace('NASDAQ:','').replace('NYSE:','') \
                          .replace('AMEX:','').replace('LSE:','') \
                          .replace('SIX:','').replace('TSX:','').replace('$','')
        if 1 <= len(raw) <= 6 and raw.isupper():
            matches.add(raw)
    return sorted(matches)

def _sentiment_score(text: str) -> float:
    words = re.findall(r'[A-Za-z]+', text.lower())
    if not words: 
        return 0.0
    score = sum(1 for w in words if w in FIN_BULLISH) - sum(1 for w in words if w in FIN_BEARISH)
    return round(score / max(100, len(words)), 4)

def _key_sentences(paragraphs: List[str], top_n: int = 5) -> List[str]:
    candidates = []
    for p in paragraphs:
        sents = SENT_SPLIT.split(p)
        if sents:
            first = sents[0]
            score = 0
            if PERCENT_RE.search(p): score += 2
            if any(k in p.lower() for k in ['guidance','earnings','forecast','revenue','profit']):
                score += 1
            candidates.append((score, first))
    candidates.sort(key=lambda x: (-x[0], len(x[1])))
    return [c[1] for c in candidates[:top_n]]

# -------------------------
# Main parser
# -------------------------
def parse_news_from_html(html: str, url: Optional[str] = None) -> NewsRecord:
    soup = BeautifulSoup(html, 'lxml')
    meta = _extract_meta(soup)
    paragraphs = _extract_body_bs4(soup)

    if _HAS_DOCLING:
        try:
            import tempfile, os
            conv = DocumentConverter()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tf:
                tf.write(html)
                temp_path = tf.name
            md = conv.convert(temp_path).document.export_to_markdown()
            os.unlink(temp_path)
            bullets = [ln.strip() for ln in md.splitlines() if ln.strip().startswith(('•','- '))]
            for b in bullets:
                if b not in paragraphs and len(b) > 5:
                    paragraphs.insert(0, b)
        except Exception:
            pass

    body = "\n\n".join(paragraphs)
    tickers = _extract_tickers(' '.join([meta.get('title') or '', body]))
    prices = PRICE_RE.findall(body)
    percents = PERCENT_RE.findall(body)
    keys = _key_sentences(paragraphs)
    senti = _sentiment_score(body)

    return NewsRecord(
        url=url,
        title=meta.get('title') or '',
        author=meta.get('author'),
        published=meta.get('published'),
        tickers=tickers,
        prices=prices,
        percents=percents,
        key_sentences=keys,
        sentiment=senti,
        body=body,
        source=meta.get('source')
    )

# -------------------------
# Export helpers
# -------------------------
def to_jsonl(records: List[NewsRecord], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

def to_csv(records: List[NewsRecord], path: str):
    import csv
    keys = list(asdict(records[0]).keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

# -------------------------
# CLI usage
# -------------------------
if __name__ == "__main__":
    import sys, pathlib
    if len(sys.argv) < 2:
        print("Usage: python trading_news_parser.py <html_file1> [<html_file2> ...]")
        sys.exit(0)
    recs = []
    for p in sys.argv[1:]:
        html = pathlib.Path(p).read_text(encoding='utf-8', errors='ignore')
        recs.append(parse_news_from_html(html, url=p))
    out = "news.jsonl"
    to_jsonl(recs, out)
    print(f"Wrote {len(recs)} records to {out}")
