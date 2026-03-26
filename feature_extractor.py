import re
import math
import ipaddress
from collections import Counter
from urllib.parse import urlparse, parse_qs
import pandas as pd


SUSPICIOUS_EXTENSIONS = {
    ".exe", ".zip", ".rar", ".scr", ".bat", ".cmd", ".js", ".jar",
    ".msi", ".php", ".asp", ".aspx", ".jsp"
}


def ensure_scheme(url: str) -> str:
    """Add https:// if scheme is missing."""
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        return "https://" + url
    return url


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def extract_domain_parts(netloc: str):
    """
    Returns:
        subdomain_count, subdomain, domain_name, root_domain, tld
    """
    if not netloc:
        return 0, "", "", "", ""

    host = netloc.split(":")[0].lower()
    parts = host.split(".")

    if len(parts) >= 2:
        tld = parts[-1]
        domain_name = parts[-2]
        subdomain_parts = parts[:-2]
        subdomain = ".".join(subdomain_parts) if subdomain_parts else ""
        root_domain = f"{domain_name}.{tld}"
        subdomain_count = len(subdomain_parts)
    else:
        tld = ""
        domain_name = host
        subdomain = ""
        root_domain = host
        subdomain_count = 0

    return subdomain_count, subdomain, domain_name, root_domain, tld


def has_ip_address(host: str) -> int:
    """Return 1 if host is an IP address, else 0."""
    if not host:
        return 0
    host = host.split(":")[0]
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def has_suspicious_extension(path: str) -> int:
    """Return 1 if path ends with a suspicious extension."""
    if not path:
        return 0
    path = path.lower()
    return int(any(path.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS))


def extract_url_features(url: str) -> dict:
    """Extract reproducible URL-based features from a single URL."""
    original_url = url if isinstance(url, str) else ""
    normalized_url = ensure_scheme(original_url)
    parsed = urlparse(normalized_url)

    full_url = normalized_url
    host = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""


    subdomain_count, subdomain, domain_name, root_domain, tld = extract_domain_parts(host)

    num_digits = sum(char.isdigit() for char in full_url)
    url_length = len(full_url)
    numeric_ratio = (num_digits / url_length * 100) if url_length > 0 else 0.0

    token_count = len([tok for tok in re.split(r"[^A-Za-z0-9]+", full_url) if tok])
    query_param_count = len(parse_qs(query)) if query else 0


    features = {
    "URL": original_url,
    "host": host,
    "subdomain": subdomain,
    "root_domain": root_domain,
    "tld": tld,
    "url_length": url_length,
    "has_ip_address": has_ip_address(host),
    "dot_count": full_url.count("."),
    "https_flag": int(parsed.scheme.lower() == "https"),
    "url_entropy": shannon_entropy(full_url),
    "token_count": token_count,
    "subdomain_count": subdomain_count,
    "query_param_count": query_param_count,
    "tld_length": len(tld),
    "path_length": len(path),
    "has_hyphen_in_domain": int("-" in domain_name),
    "number_of_digits": num_digits,
    "suspicious_file_extension": has_suspicious_extension(path),
    "domain_name_length": len(domain_name),
    "percentage_numeric_chars": numeric_ratio,
}

    return features


def extract_features_from_list(urls):
    """Extract features for a list of URLs and return a DataFrame."""
    rows = [extract_url_features(url) for url in urls]
    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]