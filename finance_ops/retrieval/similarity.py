"""Hybrid Generalized Entity Matching & Similarity Metric Scoring Engine.

Grounded in:
- Zabolotnia, T. M., & Kozynets, N. V. (2025). Hybrid detection of fuzzy duplicate texts, AAIT.
- Wang, J., Li, Y., Hirota, W., & Kandogan, E. (Megagon Labs, 2022). Machop: an End-to-End Generalized Entity Matching Framework, ACM aiDM.
- Smith, C., Sesodia, M., Lindenberg, F., & Schroeder de Witt, C. (2025). OpenSanctions Pairs: Large-Scale Entity Matching with LLMs, ICML.
"""

from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime

from finance_ops.core.models import CanonicalTransaction


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Computes Jaro-Winkler string similarity between two strings."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    t = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            t += 1
        k += 1
    transpositions = t / 2

    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    # Winkler prefix bonus
    prefix_len = 0
    for i in range(min(len1, len2, max_l)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    jaro_winkler = jaro + (prefix_len * p * (1.0 - jaro))
    return float(min(1.0, max(0.0, jaro_winkler)))


def token_set_ratio(s1: str, s2: str) -> float:
    """Computes Token Set Ratio similarity (invariance to word ordering and subsets)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0

    intersection = t1 & t2
    if not intersection:
        return 0.0

    sorted_intersect = " ".join(sorted(list(intersection)))
    sorted_t1 = " ".join(sorted(list(t1)))
    sorted_t2 = " ".join(sorted(list(t2)))

    # Compute pairwise Jaro-Winkler across combinations
    sim1 = jaro_winkler_similarity(sorted_intersect, sorted_t1)
    sim2 = jaro_winkler_similarity(sorted_intersect, sorted_t2)
    sim3 = jaro_winkler_similarity(sorted_t1, sorted_t2)

    return float(max(sim1, sim2, sim3))


def longest_common_subsequence_ratio(s1: str, s2: str) -> float:
    """Computes normalized LCS ratio for abbreviation and concatenation invariance."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    max_len = max(m, n)
    return float(lcs_len / max_len) if max_len > 0 else 0.0


def calculate_lexical_similarity(str1: str, str2: str) -> float:
    """Computes lexical token overlap similarity."""
    if not str1 or not str2:
        return 0.0
    s1, s2 = str1.lower().strip(), str2.lower().strip()
    if s1 == s2:
        return 1.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    return float(jaccard)


def calculate_name_similarity(name1: Optional[str], name2: Optional[str]) -> float:
    """
    Hybrid string similarity combining Jaro-Winkler, Token Set Ratio, and LCS
    as formulated in Zabolotnia & Kozynets (2025).
    """
    if not name1 and not name2:
        return 1.0
    if not name1 or not name2:
        return 0.0

    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return 1.0

    jw = jaro_winkler_similarity(n1, n2)
    tsr = token_set_ratio(n1, n2)
    lcs = longest_common_subsequence_ratio(n1, n2)

    score = 0.40 * jw + 0.40 * tsr + 0.20 * lcs
    return float(min(1.0, max(0.0, score)))


def calculate_amount_similarity(
    amt1_paise: int,
    amt2_paise: int,
    fee_paise: int = 0
) -> Tuple[float, bool]:
    """
    Computes piecewise amount similarity with fee-adjustment and rounding detection.
    Returns (score, is_fee_candidate).
    """
    if amt1_paise == amt2_paise:
        return 1.0, False

    diff = abs(amt1_paise - amt2_paise)
    max_amt = max(abs(amt1_paise), abs(amt2_paise))

    if max_amt == 0:
        return 1.0, False

    # Rounding difference within ₹1.00 (100 paise)
    if diff <= 100:
        return 0.95, False

    # Check for known fee rate (e.g. 2% MDR: 200 bps)
    expected_fee_2pct = int(round(max_amt * 0.02))
    if abs(diff - expected_fee_2pct) <= 200 or (fee_paise > 0 and abs(diff - fee_paise) <= 100):
        return 0.90, True

    # FX rounding / minor drift within 0.5%
    rel_diff = diff / max_amt
    if rel_diff <= 0.005:
        return 0.85, False

    # Discrepancy > 2% without fee explanation
    if rel_diff > 0.02:
        return 0.0, False

    # Smooth decay between 0.5% and 2.0%
    decay_score = 0.85 - ((rel_diff - 0.005) / 0.015) * 0.85
    return float(max(0.0, decay_score)), False


def calculate_reference_similarity(ref1: Optional[str], ref2: Optional[str]) -> float:
    """
    Computes reference identifier similarity.
    Exact match: 1.0; Prefix/Suffix: 0.90; Missing: 0.50 (neutral); Conflicting: 0.0.
    """
    if not ref1 and not ref2:
        return 0.50  # Both missing -> neutral
    if not ref1 or not ref2:
        return 0.50  # One missing -> neutral

    r1 = ref1.upper().strip()
    r2 = ref2.upper().strip()

    if r1 == r2:
        return 1.0
    if len(r1) >= 4 and len(r2) >= 4:
        if r1 in r2 or r2 in r1:
            return 0.90
        return 0.0  # Explicit conflicting references

    return 0.0


def calculate_date_similarity(ts1: int, ts2: int) -> float:
    """
    Computes temporal decay score based on RBI settlement lags (T+0, T+1, T+2, T+7).
    """
    if ts1 == 0 or ts2 == 0:
        return 0.80  # Default neutral when timestamp missing

    delta_sec = abs(ts1 - ts2)
    delta_days = delta_sec / 86400.0

    if delta_days <= 0.5:
        return 1.00
    elif delta_days <= 1.5:
        return 0.90  # T+1 UPI / Card
    elif delta_days <= 3.5:
        return 0.75  # T+2 / T+3 NetBanking / NEFT
    elif delta_days <= 7.5:
        return 0.50  # Late settlement / weekend / holiday
    else:
        return float(max(0.0, 0.50 - 0.05 * (delta_days - 7.5)))


def calculate_candidate_similarity(
    query_tx: CanonicalTransaction,
    candidate_tx: CanonicalTransaction
) -> Dict[str, Any]:
    """
    Computes full composite similarity vector between two canonical transactions
    with hard zero constraint enforcement (Smith et al. 2025).
    """
    # 1. Amount
    amt_score, is_fee_candidate = calculate_amount_similarity(
        query_tx.amount_paise,
        candidate_tx.amount_paise,
        fee_paise=query_tx.fee_paise or candidate_tx.fee_paise
    )

    # 2. Reference ID (checks UTR, Order ID, Invoice Ref, Payment Ref)
    ref_scores = []
    if query_tx.utr and candidate_tx.utr:
        ref_scores.append(1.0 if query_tx.utr.strip().upper() == candidate_tx.utr.strip().upper() else 0.0)
    if query_tx.order_id and candidate_tx.order_id:
        ref_scores.append(calculate_reference_similarity(query_tx.order_id, candidate_tx.order_id))
    if query_tx.invoice_reference and candidate_tx.invoice_reference:
        ref_scores.append(calculate_reference_similarity(query_tx.invoice_reference, candidate_tx.invoice_reference))
    if query_tx.payment_reference and candidate_tx.payment_reference:
        ref_scores.append(calculate_reference_similarity(query_tx.payment_reference, candidate_tx.payment_reference))

    id_score = max(ref_scores) if ref_scores else 0.50

    # 3. Merchant / Name
    name_score = calculate_name_similarity(
        query_tx.merchant_name_norm or query_tx.merchant_name,
        candidate_tx.merchant_name_norm or candidate_tx.merchant_name
    )

    # 4. Date
    date_score = calculate_date_similarity(query_tx.txn_timestamp, candidate_tx.txn_timestamp)

    # 5. Currency Check
    curr_score = 1.0 if query_tx.currency.upper() == candidate_tx.currency.upper() else 0.0

    # 6. Composite Score Formula: 0.40 * S_amt + 0.30 * S_ref + 0.15 * S_name + 0.15 * S_date
    composite = (
        0.40 * amt_score +
        0.30 * id_score +
        0.15 * name_score +
        0.15 * date_score
    ) * curr_score

    # Critical Hard Constraint:
    # If S_amt == 0 and S_ref == 0 -> Composite = 0.0
    if amt_score == 0.0 and id_score == 0.0:
        composite = 0.0

    return {
        "amount_score": round(amt_score, 4),
        "identifier_score": round(id_score, 4),
        "name_score": round(name_score, 4),
        "date_score": round(date_score, 4),
        "currency_score": round(curr_score, 4),
        "composite_score": round(float(composite), 4),
        "is_fee_candidate": is_fee_candidate
    }
