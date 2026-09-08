"""
Lightweight query parsing, no heavy NLP dependency required.
Matches country and region (admin) names against the known list via
simple substring/fuzzy matching, and classifies rough intent by keyword.
Good enough for a focused domain (6 countries, known region names) without
needing a full NLU stack.
"""
import difflib
import re


def _best_match(text: str, candidates: list, cutoff: float = 0.6):
    text_lower = text.lower()
    # exact substring match first (most reliable)
    for c in candidates:
        if c.lower() in text_lower:
            return c
    # fuzzy match on individual words as a fallback
    words = re.findall(r"[a-zA-Z]+", text)
    best, best_score = None, 0.0
    for c in candidates:
        for w in words:
            score = difflib.SequenceMatcher(None, w.lower(), c.lower()).ratio()
            if score > best_score:
                best, best_score = c, score
    return best if best_score >= cutoff else None


def parse_query(query: str, regions_df) -> dict:
    countries = regions_df["country"].unique().tolist()
    admins = regions_df["admin"].unique().tolist()

    matched_country = _best_match(query, countries)
    matched_admin = _best_match(query, admins)

    region_id = None
    if matched_admin:
        sub = regions_df[regions_df.admin == matched_admin]
        if matched_country:
            sub = sub[sub.country == matched_country]
        if len(sub):
            row = sub.iloc[0]
            region_id = int(row.region_id)
            matched_country = row.country

    if region_id is None and matched_country:
        # fall back to the country's most-surveyed region as a representative point
        sub = regions_df[regions_df.country == matched_country]
        if len(sub):
            region_id = int(sub.iloc[0].region_id)

    intent = "explain" if re.search(r"\bwhy\b|\breason|\bexplain", query, re.I) else "status"
    if re.search(r"\bcompare|\bversus|\bvs\b", query, re.I):
        intent = "compare"

    return {
        "country": matched_country,
        "region_id": region_id,
        "matched_admin": matched_admin,
        "intent": intent,
        "raw_query": query,
    }
