"""Deterministic Financial Invariants and Mathematical Conservation Assertions."""

from decimal import Decimal, ROUND_HALF_EVEN
from typing import List, Tuple, Optional
from finance_ops.core.models import CanonicalTransaction, TransactionType


ZERO = Decimal("0.00")
CENT = Decimal("0.01")


def quantize_amount(val: Decimal) -> Decimal:
    """Rounds to exact two decimal currency cents using Banker's Rounding (ROUND_HALF_EVEN)."""
    return val.quantize(CENT, rounding=ROUND_HALF_EVEN)


def check_amount_conservation(
    gross_amount: Decimal,
    net_amount: Decimal,
    fee_amount: Decimal,
    tax_amount: Decimal = ZERO,
    tolerance: Decimal = Decimal("0.02")
) -> Tuple[bool, Decimal]:
    """
    Verifies that: Gross = Net + Fees + Taxes (within numeric tolerance).
    Returns (is_conserved, discrepancy_amount).
    """
    expected_gross = quantize_amount(net_amount + fee_amount + tax_amount)
    actual_gross = quantize_amount(gross_amount)
    discrepancy = abs(actual_gross - expected_gross)
    is_valid = discrepancy <= tolerance
    return is_valid, discrepancy


def check_split_payment_conservation(
    parent_amount: Decimal,
    child_amounts: List[Decimal],
    tolerance: Decimal = Decimal("0.02")
) -> Tuple[bool, Decimal]:
    """
    Verifies that sum of child split payments equals parent amount.
    """
    total_children = quantize_amount(sum(child_amounts, ZERO))
    parent_quantized = quantize_amount(parent_amount)
    discrepancy = abs(parent_quantized - total_children)
    is_valid = discrepancy <= tolerance
    return is_valid, discrepancy


def check_currency_consistency(records: List[CanonicalTransaction]) -> bool:
    """Verifies all records share identical ISO-4217 currency code."""
    if not records:
        return True
    first_currency = records[0].currency.upper()
    return all(r.currency.upper() == first_currency for r in records)


def check_reversal_integrity(
    original_tx: CanonicalTransaction,
    reversal_tx: CanonicalTransaction
) -> Tuple[bool, str]:
    """
    Verifies that reversal transaction correctly cancels original transaction:
    - Amounts match in magnitude
    - Reversal timestamp is >= original timestamp
    - Currencies match
    """
    if original_tx.currency != reversal_tx.currency:
        return False, f"Currency mismatch: {original_tx.currency} vs {reversal_tx.currency}"
    
    if quantize_amount(abs(original_tx.amount)) != quantize_amount(abs(reversal_tx.amount)):
        return False, f"Reversal amount {reversal_tx.amount} does not match original {original_tx.amount}"
    
    if reversal_tx.transaction_timestamp < original_tx.transaction_timestamp:
        return False, f"Reversal date {reversal_tx.transaction_timestamp} cannot precede original {original_tx.transaction_timestamp}"
        
    return True, "Reversal integrity verified"
