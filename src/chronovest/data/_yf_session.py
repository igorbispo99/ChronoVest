"""Browser-impersonating HTTP session for Yahoo Finance.

Yahoo rejects requests that do not look like a real browser with HTTP 406
("Not Acceptable"). A curl_cffi session impersonating Chrome sends the headers
and TLS fingerprint Yahoo expects, which is the canonical fix for the 406/429
errors. Returns None if curl_cffi is unavailable, in which case recent yfinance
versions handle impersonation internally anyway.
"""

from __future__ import annotations


def make_impersonated_session():
    try:
        from curl_cffi import requests as curl_requests

        return curl_requests.Session(impersonate="chrome")
    except Exception:  # curl_cffi missing or incompatible
        return None
