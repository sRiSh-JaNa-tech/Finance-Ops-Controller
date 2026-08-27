"""Combinatorial 1-to-Many & Many-to-One Split Payment Constraint Solver."""

from typing import List, Dict, Tuple, Optional, Any, Set
from decimal import Decimal
from datetime import datetime
from finance_ops.core.models import CanonicalTransaction
from finance_ops.core.invariants import quantize_amount


# Registry of already-reconciled transaction IDs — prevents double-counting.
_RECONCILED_IDS: Set[str] = set()


def mark_reconciled(transaction_ids: List[str]) -> None:
    """Marks transaction IDs as reconciled so they cannot be used again."""
    _RECONCILED_IDS.update(transaction_ids)


def is_already_reconciled(transaction_id: str) -> bool:
    """Returns True if a transaction has already been matched in a previous case."""
    return transaction_id in _RECONCILED_IDS


def reset_reconciliation_registry() -> None:
    """Clears the reconciliation registry (used between benchmark seeds)."""
    _RECONCILED_IDS.clear()


class SplitReconciliationSolver:
    """
    Solves 1:N and N:1 subset-sum combinatorial reconciliation constraints.

    1-to-N: Finds a subset of candidate transactions {C_j} such that:
        sum(Amount(C_j) - Fee(C_j)) ≈ TargetAmount  within tolerance

    N-to-1: Finds a subset of invoice amounts {I_j} such that:
        sum(Amount(I_j)) ≈ SettlementGrossAmount  within tolerance

    Penalty terms:
        - Temporal dispersion: penalize wide date spread across matched children
        - Entity mismatch: penalize if customer IDs differ across matched children
    """

    def __init__(
        self,
        tolerance: Decimal = Decimal("0.02"),
        max_subset_size: int = 5,
        temporal_penalty_weight: float = 0.05,
        entity_penalty_weight: float = 0.10
    ):
        self.tolerance = tolerance
        self.max_subset_size = max_subset_size
        self.temporal_penalty_weight = temporal_penalty_weight
        self.entity_penalty_weight = entity_penalty_weight

    def _compute_penalty(
        self,
        subset: List[CanonicalTransaction],
        reference_tx: CanonicalTransaction
    ) -> float:
        """
        Computes combined penalty for temporal dispersion and entity mismatch.

        Temporal dispersion penalty: proportional to std-dev of date offsets from reference.
        Entity mismatch penalty: fraction of subset members with different customer_id.
        """
        if not subset:
            return 0.0

        ref_ts = reference_tx.transaction_timestamp
        offsets_days = [
            abs((c.transaction_timestamp - ref_ts).total_seconds()) / 86400.0
            for c in subset
        ]
        # Temporal dispersion: mean offset in days normalized by 30-day window
        mean_offset = sum(offsets_days) / len(offsets_days)
        temporal_penalty = self.temporal_penalty_weight * min(mean_offset / 30.0, 1.0)

        # Entity mismatch: fraction of children that disagree with reference customer
        ref_cust = reference_tx.customer_id
        if ref_cust:
            mismatched = sum(
                1 for c in subset
                if c.customer_id and c.customer_id != ref_cust
            )
            entity_penalty = self.entity_penalty_weight * (mismatched / len(subset))
        else:
            entity_penalty = 0.0

        return temporal_penalty + entity_penalty

    def solve_1_to_n(
        self,
        target_tx: CanonicalTransaction,
        candidate_pool: List[CanonicalTransaction]
    ) -> Optional[Dict[str, Any]]:
        """
        Finds a subset of transactions in candidate_pool that sums to target_tx.amount.

        Implements: min |TargetAmount - sum(Amount_j - Fee_j)| subject to x_j ∈ {0,1}, Δt_j ≤ τ
        Uses backtracking subset-sum with temporal proximity sorting and double-counting guard.

        Returns solution dict with matched child IDs, discrepancy, penalty, and sum amount if found.
        """
        target_amt = quantize_amount(target_tx.amount)

        # Exclude already-reconciled records
        available = [c for c in candidate_pool if not is_already_reconciled(c.transaction_id)]

        # Sort candidates by date proximity to target (closest first) and cap at 15
        sorted_candidates = sorted(
            available,
            key=lambda c: abs((c.transaction_timestamp - target_tx.transaction_timestamp).total_seconds())
        )[:15]

        best_subset: Optional[List[CanonicalTransaction]] = None
        min_score = float("inf")  # score = discrepancy + penalty

        def backtrack(index: int, current_subset: List[CanonicalTransaction], current_sum: Decimal):
            nonlocal best_subset, min_score

            discrepancy = abs(target_amt - current_sum)
            if len(current_subset) >= 2 and discrepancy <= self.tolerance:
                penalty = self._compute_penalty(current_subset, target_tx)
                score = float(discrepancy) + penalty
                if score < min_score:
                    min_score = score
                    best_subset = list(current_subset)
                return

            if len(current_subset) >= self.max_subset_size or index >= len(sorted_candidates):
                return

            cand = sorted_candidates[index]
            # Net amount: amount minus any processor fee
            net_amt = quantize_amount(cand.amount - cand.fee)

            # Prune: skip if adding this candidate would exceed target by more than tolerance
            if current_sum + net_amt <= target_amt + self.tolerance:
                current_subset.append(cand)
                backtrack(index + 1, current_subset, current_sum + net_amt)
                current_subset.pop()

            # Branch: exclude
            backtrack(index + 1, current_subset, current_sum)

        backtrack(0, [], Decimal("0.00"))

        if best_subset and float(abs(target_amt - sum(
            quantize_amount(c.amount - c.fee) for c in best_subset
        ))) <= float(self.tolerance) + 0.001:
            penalty = self._compute_penalty(best_subset, target_tx)
            return {
                "direction": "1:N",
                "target_transaction_id": target_tx.transaction_id,
                "target_amount": float(target_amt),
                "matched_transaction_ids": [c.transaction_id for c in best_subset],
                "subset_sum": float(sum((quantize_amount(c.amount - c.fee) for c in best_subset), Decimal("0.00"))),
                "discrepancy": float(abs(target_amt - sum(
                    quantize_amount(c.amount - c.fee) for c in best_subset
                ))),
                "penalty_score": round(penalty, 4),
                "child_count": len(best_subset)
            }
        return None

    def solve_n_to_1(
        self,
        settlement_tx: CanonicalTransaction,
        invoice_pool: List[CanonicalTransaction]
    ) -> Optional[Dict[str, Any]]:
        """
        Many-to-One: Finds a subset of invoices {I_j} whose gross amounts sum to
        the settlement batch's gross_volume.

        Implements: min |SettlementGross - sum(Amount_j)| subject to x_j ∈ {0,1}, Δt_j ≤ τ

        Returns solution dict if a valid subset is found.
        """
        target_amt = quantize_amount(settlement_tx.amount)

        # Exclude already-reconciled
        available = [c for c in invoice_pool if not is_already_reconciled(c.transaction_id)]

        # Sort by closest amount (for heuristic pruning) and cap at 20
        sorted_pool = sorted(available, key=lambda c: float(c.amount), reverse=True)[:20]

        best_subset: Optional[List[CanonicalTransaction]] = None
        min_score = float("inf")

        def backtrack(index: int, current_subset: List[CanonicalTransaction], current_sum: Decimal):
            nonlocal best_subset, min_score

            discrepancy = abs(target_amt - current_sum)
            if len(current_subset) >= 1 and discrepancy <= self.tolerance:
                penalty = self._compute_penalty(current_subset, settlement_tx)
                score = float(discrepancy) + penalty
                if score < min_score:
                    min_score = score
                    best_subset = list(current_subset)

            if len(current_subset) >= self.max_subset_size or index >= len(sorted_pool):
                return

            cand = sorted_pool[index]
            cand_amt = quantize_amount(cand.amount)

            # Prune: skip if overshoots
            if current_sum + cand_amt <= target_amt + self.tolerance:
                current_subset.append(cand)
                backtrack(index + 1, current_subset, current_sum + cand_amt)
                current_subset.pop()

            backtrack(index + 1, current_subset, current_sum)

        backtrack(0, [], Decimal("0.00"))

        if best_subset:
            subset_sum = sum((quantize_amount(c.amount) for c in best_subset), Decimal("0.00"))
            disc = float(abs(target_amt - subset_sum))
            if disc <= float(self.tolerance) + 0.001:
                penalty = self._compute_penalty(best_subset, settlement_tx)
                return {
                    "direction": "N:1",
                    "settlement_transaction_id": settlement_tx.transaction_id,
                    "settlement_amount": float(target_amt),
                    "matched_transaction_ids": [c.transaction_id for c in best_subset],
                    "subset_sum": float(subset_sum),
                    "discrepancy": disc,
                    "penalty_score": round(penalty, 4),
                    "invoice_count": len(best_subset)
                }
        return None

