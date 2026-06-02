#!/usr/bin/env python3
"""
probe_targets.py — Lightweight check of the 6 maison scraper targets.

Does NOT run Selenium (saves hours). Instead, fetches each landing page with
requests + a realistic User-Agent and checks:

  1. HTTP status (200 = site still serves the URL)
  2. Title / canonical (sanity check — same brand)
  3. Whether the key product CSS class from the scraper is present in raw HTML

If a class is missing, the scraper is almost certainly broken (the site
re-built its CSS class names). If the class is present, the scraper has a
chance — but Selenium-level validation is still needed since most luxury
sites are heavy JS rendering.

Run:
    python probe_targets.py
"""

import re
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

TARGETS = [
    {
        "brand": "Prada",
        "url":   "https://www.prada.com/it/it/mens/accessories/c/10156EU",
        "probes": [
            ("product list item",  r'li class="[^"]*\bw-full\b'),
            ("product picture",    r'picture class="[^"]*\bstd-small'),
            ("price tag",          r'product-card__price'),
        ],
    },
    {
        "brand": "Cartier",
        "url":   "https://www.cartier.com/it-it/bags-and-accessories/sunglasses?page=0&srule=recommended",
        "probes": [
            ("product div",   r'class="[^"]*\bproduct\b'),
            ("product image", r'class="[^"]*\bproduct__image\b'),
        ],
    },
    {
        "brand": "Fendi",
        "url":   "https://www.fendi.com/it-it/search?q=occhiali&lang=it_IT",
        "probes": [
            ("product div",      r'class="[^"]*\bproduct\b'),
            ("image container",  r'class="[^"]*\bimage-container\b'),
            ("load more button", r'class="[^"]*\bload-more-btn\b'),
        ],
    },
    {
        "brand": "Bottega Veneta",
        "url":   "https://www.bottegaveneta.com/it-it/search?q=occhiali",
        "probes": [
            ("article.c-product",      r'<article[^>]*class="[^"]*\bc-product\b'),
            ("price c-price__value",   r'\bc-price__value--current\b'),
            ("data-pid attribute",     r'data-pid='),
        ],
    },
    {
        "brand": "Dolce & Gabbana",
        "url":   "https://www.dolcegabbana.com/it-it/moda/uomo/occhiali-da-sole/",
        "probes": [
            ("SearchHitsItem",   r'SearchHitsItem__search-hit'),
            ("ProductMedia",     r'ProductMedia__product-media__image-wrapper'),
            ("ProductPrice",     r'ProductPriceDiscount__product-price'),
        ],
    },
    {
        "brand": "YSL",
        "url":   "https://www.ysl.com/it-it/search?q=occhiali%20da%20sole",
        "probes": [
            ("grid-item li",  r'id="grid-item-\d+"'),
            ("data-product",  r'data-product'),
        ],
    },
]


class TitleFinder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and self.title is None:
            self.title = data.strip()


def fetch(url: str, timeout: int = 20) -> tuple[int, str, dict]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body, dict(resp.headers.items())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, body, dict(e.headers.items() if e.headers else {})


def probe_one(target: dict) -> dict:
    t0 = time.time()
    try:
        status, body, headers = fetch(target["url"], timeout=15)
    except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
        return {
            "brand": target["brand"], "url": target["url"],
            "status": "TIMEOUT/ERR", "title": str(e)[:100], "body_bytes": 0,
            "elapsed_s": round(time.time() - t0, 2), "content_type": "?",
            "redirect_to": "", "probes": [(n, False) for n, _ in target["probes"]],
        }
    elapsed = time.time() - t0

    parser = TitleFinder()
    try:
        parser.feed(body)
    except Exception:
        pass

    probe_results = []
    for name, pattern in target["probes"]:
        match = bool(re.search(pattern, body))
        probe_results.append((name, match))

    return {
        "brand": target["brand"],
        "url": target["url"],
        "status": status,
        "title": (parser.title or "")[:100],
        "body_bytes": len(body),
        "elapsed_s": round(elapsed, 2),
        "content_type": headers.get("Content-Type", "?"),
        "redirect_to": headers.get("Location", ""),
        "probes": probe_results,
    }


def main():
    print(f"{'brand':18s} {'status':>6s} {'size':>8s} {'time':>6s}  hits / total")
    print("-" * 80)
    summary = []
    for target in TARGETS:
        r = probe_one(target)
        hits = sum(1 for _, ok in r["probes"] if ok)
        total = len(r["probes"])
        print(f"{r['brand']:18s} {r['status']:>6} {r['body_bytes']:>8d} {r['elapsed_s']:>6.2f}  {hits}/{total}")
        print(f"  title: {r['title']}")
        for name, ok in r["probes"]:
            tag = "✓" if ok else "✗"
            print(f"    [{tag}] {name}")
        if r["redirect_to"]:
            print(f"  redirected → {r['redirect_to']}")
        print()
        summary.append((r["brand"], r["status"], hits, total, r["body_bytes"]))

    print("=" * 80)
    print("VERDICT (static-HTML probe — not a full Selenium run)")
    print("=" * 80)
    for brand, status, hits, total, body_bytes in summary:
        if status == "TIMEOUT/ERR":
            print(f"  {brand:18s}  ✗ network error or timeout — re-probe needed")
        elif status == 403:
            print(f"  {brand:18s}  ⚠ HTTP 403 — bot-blocked at the edge (Akamai/Cloudflare). "
                  f"Selenium with real Chrome should still work")
        elif status == 404 or status == 410:
            print(f"  {brand:18s}  ✗ HTTP {status} — URL itself is gone, need new entry point")
        elif status != 200:
            print(f"  {brand:18s}  ✗ HTTP {status} — abnormal response")
        elif hits == 0 and body_bytes > 200_000:
            print(f"  {brand:18s}  ~ HTML loads big ({body_bytes:,} bytes) but no selector matched — "
                  f"likely JS-rendered SPA, scraper may still work via Selenium "
                  f"but selectors may have changed; needs runtime check")
        elif hits == 0:
            print(f"  {brand:18s}  ✗ HTML loads but no selector matched — markup changed, scraper broken")
        elif hits < total:
            print(f"  {brand:18s}  ~ partial — {hits}/{total} selectors present, may need targeted fix")
        else:
            print(f"  {brand:18s}  ✓ all selectors present in static HTML — likely still scraping correctly")


if __name__ == "__main__":
    main()
