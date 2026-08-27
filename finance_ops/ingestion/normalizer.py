"""Lossless Financial Normalization Engine with Field-Level Provenance Attachment for Prototype 3."""

import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Tuple, Optional, List

from finance_ops.core.models import (
    SourceSystem, TransactionType, PaymentMethod, TransactionStatus,
    CanonicalTransaction, FieldProvenance
)
from finance_ops.core.invariants import quantize_amount
from finance_ops.core.provenance import create_field_provenance


# Standard Indian corporate and banking abbreviation dictionary
ABBREVIATION_MAP = {
    "pvt": "private",
    "ltd": "limited",
    "bk": "bank",
    "co": "company",
    "corp": "corporation",
    "inc": "incorporated",
    "tech": "technology",
    "technologies": "technology",
    "in": "india",
    "ind": "india",
    "svcs": "services",
    "srvc": "services",
    "mgmt": "management",
    "soln": "solutions",
    "solns": "solutions",
    "ent": "enterprise",
    "enterprises": "enterprise",
    "amzn": "amazon",
    "amznwebservices": "amazon web services",
    "flipkartt": "flipkart",
}

STOP_WORDS = {
    "payment", "for", "to", "by", "invoice", "order", "the", "a", "an",
    "on", "of", "at", "in", "via", "ref", "txn", "transfer", "neft", "rtgs", "imps", "upi"
}


def normalize_amount_paise(raw_val: Any) -> Tuple[int, Decimal, str]:
    """
    Cleans and normalizes monetary strings into exact integer paise and Decimal INR.
    Handles currency symbols (₹, Rs, $, €), negative brackets like '(100.00)', and commas.
    1 INR = 100 paise.
    """
    if raw_val is None:
        return 0, Decimal("0.00"), "null_fallback_zero"

    if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
        dec = Decimal(str(raw_val))
        paise = int(round(dec * 100))
        return paise, quantize_amount(dec), "numeric_to_paise"

    if isinstance(raw_val, Decimal):
        paise = int(round(raw_val * 100))
        return paise, quantize_amount(raw_val), "decimal_to_paise"

    raw_str = str(raw_val).strip()
    clean_str = (
        raw_str.replace("₹", "")
        .replace("Rs.", "")
        .replace("Rs", "")
        .replace("INR", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace(",", "")
        .strip()
    )

    # Handle accounting negative parenthetical format: (123.45) -> -123.45
    if clean_str.startswith("(") and clean_str.endswith(")"):
        clean_str = "-" + clean_str[1:-1]

    try:
        dec = Decimal(clean_str)
        paise = int(round(dec * 100))
        return paise, quantize_amount(dec), "regex_clean_and_paise_quantize"
    except (InvalidOperation, ValueError):
        return 0, Decimal("0.00"), "fallback_zero_on_invalid_amount"


def normalize_amount(raw_val: Any) -> Tuple[Decimal, str]:
    """Compatibility wrapper returning (Decimal, str)."""
    _, dec, trans = normalize_amount_paise(raw_val)
    return dec, trans


def normalize_merchant_name(raw_val: Any) -> Tuple[str, str]:
    """
    Cleans merchant strings: converts to lowercase, strips punctuation,
    expands standard corporate abbreviations, and collapses whitespace.
    """
    if not raw_val:
        return "", "empty_merchant_name"

    raw_str = str(raw_val).lower()
    cleaned = re.sub(r"[^\w\s]", " ", raw_str)
    tokens = cleaned.split()

    expanded_tokens = [ABBREVIATION_MAP.get(token, token) for token in tokens]
    normalized_name = " ".join(expanded_tokens).strip()

    return normalized_name, "abbreviation_expanded_and_cleaned"


def normalize_timestamp(raw_val: Any) -> Tuple[int, datetime, str]:
    """
    Normalizes multiple date/time formats into Unix epoch seconds and timezone-aware UTC datetime.
    Supports ISO formats, standard strings, dates without time, and integer timestamps.
    """
    if raw_val is None:
        epoch = 0
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return epoch, dt, "epoch_fallback_on_null"

    if isinstance(raw_val, (int, float)):
        epoch = int(raw_val)
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return epoch, dt, "timestamp_direct_epoch"

    if isinstance(raw_val, datetime):
        if raw_val.tzinfo is None:
            dt = raw_val.replace(tzinfo=timezone.utc)
        else:
            dt = raw_val.astimezone(timezone.utc)
        return int(dt.timestamp()), dt, "datetime_to_utc_epoch"

    raw_str = str(raw_val).strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%b-%Y",
        "%d-%b-%y",
        "%d %b %Y",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(raw_str, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            dt = parsed.astimezone(timezone.utc)
            return int(dt.timestamp()), dt, f"parsed_strftime_{fmt}"
        except ValueError:
            continue

    epoch_default = 0
    dt_default = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return epoch_default, dt_default, "epoch_fallback_on_parse_failure"


def normalize_reference_id(raw_val: Any) -> Tuple[str, str]:
    """
    Sanitizes reference identifiers: removes common prefixes (ORD-, INV-, UPI-, TXN-),
    dashes, spaces, slashes, and leading zeros, returning clean uppercase string.
    """
    if not raw_val:
        return "", "empty_reference"

    raw_str = str(raw_val).strip().upper()
    for prefix in ["ORD-", "INV-", "UPI-", "TXN-", "REF-", "PAY-", "BATCH-", "ORDER_", "INV_"]:
        if raw_str.startswith(prefix):
            raw_str = raw_str[len(prefix):]

    clean_str = re.sub(r"[\s\-_/]", "", raw_str)
    if clean_str.isdigit():
        clean_str = clean_str.lstrip("0") or "0"

    return clean_str, "stripped_prefix_and_leading_zeros"


def normalize_description(raw_val: Any) -> Tuple[List[str], str, str]:
    """
    Tokenizes and filters narratives: removes punctuation, stop words,
    and returns both the filtered token list and normalized string.
    """
    if not raw_val:
        return [], "", "empty_description"

    raw_str = str(raw_val).lower()
    cleaned = re.sub(r"[^\w\s]", " ", raw_str)
    raw_tokens = cleaned.split()

    filtered_tokens = [t for t in raw_tokens if t not in STOP_WORDS and len(t) > 1]
    normalized_str = " ".join(filtered_tokens)

    return filtered_tokens, normalized_str, "tokenized_and_stopwords_removed"


def normalize_narrative(raw_val: Any) -> Tuple[str, str]:
    """Compatibility wrapper returning (str, str)."""
    _, desc, trans = normalize_description(raw_val)
    return desc, trans


def normalize_currency(raw_val: Any) -> Tuple[str, str]:
    """Maps currency strings/symbols to standard ISO 4217 code."""
    if not raw_val:
        return "INR", "default_inr_on_empty"

    curr_str = str(raw_val).strip().upper()
    mapping = {
        "₹": "INR",
        "RS": "INR",
        "RS.": "INR",
        "INR": "INR",
        "$": "USD",
        "USD": "USD",
        "€": "EUR",
        "EUR": "EUR",
        "£": "GBP",
        "GBP": "GBP",
    }
    res = mapping.get(curr_str, curr_str[:3])
    return res, "iso4217_normalized"


def normalize_payment_method(raw_val: Any) -> PaymentMethod:
    if not raw_val:
        return PaymentMethod.NA
    val = str(raw_val).strip().upper()
    for method in PaymentMethod:
        if method.value in val:
            return method
    return PaymentMethod.NA


def normalize_transaction_status(raw_val: Any) -> TransactionStatus:
    if not raw_val:
        return TransactionStatus.CAPTURED
    val = str(raw_val).strip().upper()
    for st in TransactionStatus:
        if st.value in val:
            return st
    return TransactionStatus.CAPTURED


def normalize_raw_transaction(
    source_system: SourceSystem,
    source_record_id: str,
    raw_payload: Dict[str, Any],
    ground_truth_tx_id: Optional[str] = None
) -> CanonicalTransaction:
    """Ingests raw dictionary and builds a CanonicalTransaction with full field-level lineage."""
    provenance: Dict[str, FieldProvenance] = {}

    # 1. Amount
    raw_amt = raw_payload.get("amount") or raw_payload.get("amount_paise") or raw_payload.get("Debit") or raw_payload.get("Credit") or "0.00"
    amt_paise, amt_dec, amt_trans = normalize_amount_paise(raw_amt)
    provenance["amount"] = create_field_provenance(source_system, "amount", str(raw_amt), str(amt_dec), amt_trans)

    # 2. Date
    raw_date = raw_payload.get("created_at") or raw_payload.get("txn_date") or raw_payload.get("date") or raw_payload.get("Date") or raw_payload.get("created_utc")
    epoch_ts, dt_utc, dt_trans = normalize_timestamp(raw_date)
    provenance["transaction_timestamp"] = create_field_provenance(source_system, "date", str(raw_date), dt_utc.isoformat(), dt_trans)

    # 3. Settlement Date
    raw_settle = raw_payload.get("settlement_date") or raw_payload.get("settled_at") or raw_payload.get("PostDate")
    settle_epoch = None
    settle_dt = None
    if raw_settle:
        settle_epoch, settle_dt, settle_trans = normalize_timestamp(raw_settle)
        provenance["settlement_timestamp"] = create_field_provenance(source_system, "settlement_date", str(raw_settle), settle_dt.isoformat(), settle_trans)

    # 4. Merchant Name
    raw_m_name = raw_payload.get("merchant_name") or raw_payload.get("merchant") or raw_payload.get("vendor") or ""
    m_name_norm, m_name_trans = normalize_merchant_name(raw_m_name)
    provenance["merchant_name"] = create_field_provenance(source_system, "merchant_name", str(raw_m_name), m_name_norm, m_name_trans)

    # 5. Narrative & Description
    raw_desc = raw_payload.get("description") or raw_payload.get("raw_narrative") or raw_payload.get("Description") or ""
    tokens, desc_norm, desc_trans = normalize_description(raw_desc)
    provenance["narrative"] = create_field_provenance(source_system, "narrative", str(raw_desc), desc_norm, desc_trans)

    # 6. References
    raw_ref = raw_payload.get("reference") or raw_payload.get("invoice_id") or raw_payload.get("invoice_ref") or raw_payload.get("order_id") or ""
    ref_norm, ref_trans = normalize_reference_id(raw_ref)
    provenance["reference"] = create_field_provenance(source_system, "reference", str(raw_ref), ref_norm, ref_trans)

    # 7. Fee & Tax
    raw_fee = raw_payload.get("fee") or raw_payload.get("fee_paise") or "0.00"
    fee_paise, fee_dec, _ = normalize_amount_paise(raw_fee)

    raw_tax = raw_payload.get("tax") or raw_payload.get("gst_paise") or "0.00"
    gst_paise, tax_dec, _ = normalize_amount_paise(raw_tax)

    net_paise = amt_paise - fee_paise - gst_paise
    net_dec = Decimal(net_paise) / Decimal(100)

    # 8. Currency
    raw_curr = raw_payload.get("currency", "INR")
    curr_norm, _ = normalize_currency(raw_curr)

    # 9. Payment Method and Status
    pm = normalize_payment_method(raw_payload.get("method") or raw_payload.get("payment_method"))
    status = normalize_transaction_status(raw_payload.get("status"))

    # 10. Specific IDs
    order_id = raw_payload.get("order_id")
    invoice_ref = raw_payload.get("invoice_reference") or raw_payload.get("invoice_id") or raw_payload.get("invoice_ref")
    utr = raw_payload.get("utr") or raw_payload.get("rrn")
    vpa = raw_payload.get("vpa")
    gstin = raw_payload.get("gstin")
    approval_code = raw_payload.get("approval_code")
    reason_code = raw_payload.get("reason_code")
    parent_txn_id = raw_payload.get("parent_txn_id") or raw_payload.get("parent_transaction_id")

    tx_type_str = str(raw_payload.get("transaction_type", "PAYMENT")).upper()
    try:
        tx_type = TransactionType(tx_type_str)
    except ValueError:
        tx_type = TransactionType.PAYMENT

    is_reversal = raw_payload.get("is_reversal", False) or (status == TransactionStatus.REVERSED)
    is_refund = raw_payload.get("is_refund", False) or (status == TransactionStatus.REFUNDED) or (tx_type == TransactionType.REFUND)
    is_split = raw_payload.get("is_split", False)

    return CanonicalTransaction(
        transaction_id=str(raw_payload.get("transaction_id", f"TXN_{source_system.value}_{source_record_id}")),
        source_system=source_system,
        source_record_id=str(source_record_id),
        ground_truth_tx_id=ground_truth_tx_id or raw_payload.get("ground_truth_tx_id"),
        amount_paise=amt_paise,
        fee_paise=fee_paise,
        gst_paise=gst_paise,
        net_paise=net_paise,
        amount=amt_dec,
        currency=curr_norm,
        fee=fee_dec,
        tax=tax_dec,
        net_amount=net_dec,
        txn_timestamp=epoch_ts,
        settlement_timestamp_epoch=settle_epoch,
        transaction_timestamp=dt_utc,
        settlement_timestamp=settle_dt,
        customer_id=raw_payload.get("customer_id"),
        customer_name=raw_payload.get("customer_name"),
        merchant_id=raw_payload.get("merchant_id"),
        merchant_name=raw_m_name or None,
        merchant_name_norm=m_name_norm or None,
        counterparty_name=raw_payload.get("counterparty_name"),
        account_number_mask=raw_payload.get("account_number_mask"),
        order_id=order_id,
        invoice_reference=invoice_ref,
        payment_reference=ref_norm or None,
        batch_reference=raw_payload.get("batch_reference"),
        utr=utr,
        vpa=vpa,
        gstin=gstin,
        approval_code=approval_code,
        reason_code=reason_code,
        raw_narrative=str(raw_desc),
        normalized_narrative=desc_norm,
        payment_method=pm,
        status=status,
        transaction_type=tx_type,
        is_reversal=is_reversal,
        is_refund=is_refund,
        is_split=is_split,
        parent_transaction_id=parent_txn_id,
        provenance=provenance,
    )
