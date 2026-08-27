"""Production Double-Entry General Ledger Engine & Journal Posting.

Guarantees strict double-entry ledger invariants:
    sum(Debits) == sum(Credits) to the exact integer paise.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field
import uuid

from finance_ops.core.models import CanonicalTransaction, FinalDecisionRecord, DecisionLabel, ReasonCode


class AccountType(str, Enum):
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"
    EQUITY = "EQUITY"
    REVENUE = "REVENUE"
    EXPENSE = "EXPENSE"


class ChartOfAccounts:
    # Standard ERP / General Ledger Account Codes
    CASH_AT_BANK = "1010"                      # ASSET (Debit normal)
    ACCOUNTS_RECEIVABLE_CLEARING = "1020"      # ASSET (Debit normal)
    GST_INPUT_TAX_RECEIVABLE = "1040"          # ASSET (Debit normal)
    PAYMENT_GATEWAY_CLEARING = "1050"          # ASSET (Debit normal)
    ACCOUNTS_PAYABLE_UNMATCHED = "2010"        # LIABILITY (Credit normal)
    DISPUTED_SETTLEMENT_SUSPENSE = "2090"      # LIABILITY (Credit normal)
    MERCHANT_PROCESSING_FEE_EXPENSE = "5020"   # EXPENSE (Debit normal)
    REVENUE_LEAKAGE_LOSS = "5090"              # EXPENSE (Debit normal)


class JournalEntryLine(BaseModel):
    account_code: str
    account_name: str
    account_type: AccountType
    debit_paise: int = 0
    credit_paise: int = 0
    narration: str = ""


class DoubleEntryJournalEntry(BaseModel):
    """
    Immutable Double-Entry General Ledger Journal Entry.
    Strictly asserts balance: sum(debit_paise) == sum(credit_paise).
    """
    entry_id: str = Field(default_factory=lambda: f"JE_{uuid.uuid4().hex[:12].upper()}")
    case_id: str
    decision_id: str
    posted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    currency: str = "INR"
    lines: List[JournalEntryLine] = Field(default_factory=list)
    total_debit_paise: int = 0
    total_credit_paise: int = 0
    is_balanced: bool = False
    reconciliation_state: str = "POSTED"

    def model_post_init(self, __context: Any) -> None:
        self.total_debit_paise = sum(line.debit_paise for line in self.lines)
        self.total_credit_paise = sum(line.credit_paise for line in self.lines)
        self.is_balanced = (self.total_debit_paise == self.total_credit_paise)
        if not self.is_balanced:
            raise ValueError(
                f"Double-entry ledger out of balance! Total Debits: {self.total_debit_paise} paise != Total Credits: {self.total_credit_paise} paise"
            )


class GeneralLedgerPostingEngine:
    """
    Translates reconciled matches and audit decisions into ledger-grade
    balanced double-entry journal entries.
    """

    @staticmethod
    def create_journal_entry(
        decision: FinalDecisionRecord,
        source_tx: CanonicalTransaction,
        target_txs: List[CanonicalTransaction]
    ) -> DoubleEntryJournalEntry:
        lines: List[JournalEntryLine] = []

        if decision.decision == DecisionLabel.MATCHED:
            # Case A: Clean Exact or Identity Match
            if decision.reason in (ReasonCode.EXACT_IDENTIFIER_MATCH, ReasonCode.FUZZY_ENTITY_MATCH, ReasonCode.TIMING_ALIGNED_MATCH, ReasonCode.GRAPH_RESOLVED_MATCH):
                lines.append(
                    JournalEntryLine(
                        account_code=ChartOfAccounts.CASH_AT_BANK,
                        account_name="Cash at Bank (Settlement)",
                        account_type=AccountType.ASSET,
                        debit_paise=source_tx.amount_paise,
                        narration=f"Settlement received for {source_tx.invoice_reference or source_tx.transaction_id}"
                    )
                )
                lines.append(
                    JournalEntryLine(
                        account_code=ChartOfAccounts.ACCOUNTS_RECEIVABLE_CLEARING,
                        account_name="Accounts Receivable Clearing",
                        account_type=AccountType.ASSET,
                        credit_paise=source_tx.amount_paise,
                        narration=f"Clear invoice receivable for {source_tx.invoice_reference or source_tx.transaction_id}"
                    )
                )

            # Case B: Fee Deducted Match (MDR + GST)
            elif decision.reason == ReasonCode.FEE_ADJUSTED_MATCH:
                target_tx = target_txs[0] if target_txs else source_tx
                fee_paise = source_tx.amount_paise - target_tx.amount_paise
                # 18% GST calculation
                gst_paise = int(round(fee_paise * 18 / 118)) if fee_paise > 0 else 0
                net_fee_paise = fee_paise - gst_paise
                settled_paise = target_tx.amount_paise

                lines.append(
                    JournalEntryLine(
                        account_code=ChartOfAccounts.CASH_AT_BANK,
                        account_name="Cash at Bank (Net Settlement)",
                        account_type=AccountType.ASSET,
                        debit_paise=settled_paise,
                        narration=f"Net settlement credited"
                    )
                )
                if net_fee_paise > 0:
                    lines.append(
                        JournalEntryLine(
                            account_code=ChartOfAccounts.MERCHANT_PROCESSING_FEE_EXPENSE,
                            account_name="MDR Merchant Processing Fee",
                            account_type=AccountType.EXPENSE,
                            debit_paise=net_fee_paise,
                            narration=f"Processor MDR fee deducted"
                        )
                    )
                if gst_paise > 0:
                    lines.append(
                        JournalEntryLine(
                            account_code=ChartOfAccounts.GST_INPUT_TAX_RECEIVABLE,
                            account_name="GST Input Tax Credit",
                            account_type=AccountType.ASSET,
                            debit_paise=gst_paise,
                            narration=f"Input GST credit on processing fee"
                        )
                    )
                lines.append(
                    JournalEntryLine(
                        account_code=ChartOfAccounts.ACCOUNTS_RECEIVABLE_CLEARING,
                        account_name="Accounts Receivable Clearing",
                        account_type=AccountType.ASSET,
                        credit_paise=source_tx.amount_paise,
                        narration=f"Clear gross invoice receivable"
                    )
                )

            # Case C: Split Payment Match
            elif decision.reason == ReasonCode.SPLIT_PAYMENT_MATCH:
                total_settled_paise = sum(t.amount_paise for t in target_txs)
                lines.append(
                    JournalEntryLine(
                        account_code=ChartOfAccounts.CASH_AT_BANK,
                        account_name="Cash at Bank (Split Tranches)",
                        account_type=AccountType.ASSET,
                        debit_paise=total_settled_paise,
                        narration=f"Split settlement tranches ({len(target_txs)} parts)"
                    )
                )
                lines.append(
                    JournalEntryLine(
                        account_code=ChartOfAccounts.ACCOUNTS_RECEIVABLE_CLEARING,
                        account_name="Accounts Receivable Clearing",
                        account_type=AccountType.ASSET,
                        credit_paise=total_settled_paise,
                        narration=f"Clear split invoice {source_tx.invoice_reference}"
                    )
                )

            # Default matched entry
            else:
                lines.append(
                    JournalEntryLine(
                        account_code=ChartOfAccounts.CASH_AT_BANK,
                        account_name="Cash at Bank",
                        account_type=AccountType.ASSET,
                        debit_paise=source_tx.amount_paise,
                        narration="Reconciled cash settlement"
                    )
                )
                lines.append(
                    JournalEntryLine(
                        account_code=ChartOfAccounts.PAYMENT_GATEWAY_CLEARING,
                        account_name="Gateway Clearing",
                        account_type=AccountType.ASSET,
                        credit_paise=source_tx.amount_paise,
                        narration="Clear gateway transit balance"
                    )
                )

        else:
            # Exceptions & Uncertain Suspense Posting
            lines.append(
                JournalEntryLine(
                    account_code=ChartOfAccounts.DISPUTED_SETTLEMENT_SUSPENSE,
                    account_name="Disputed / Unmatched Settlement Suspense",
                    account_type=AccountType.LIABILITY,
                    debit_paise=source_tx.amount_paise,
                    narration=f"Quarantined in suspense queue: {decision.reason.value}"
                )
            )
            lines.append(
                JournalEntryLine(
                    account_code=ChartOfAccounts.ACCOUNTS_PAYABLE_UNMATCHED,
                    account_name="Unmatched Transaction Holding",
                    account_type=AccountType.LIABILITY,
                    credit_paise=source_tx.amount_paise,
                    narration=f"Holding pending manual audit: {decision.decision.value}"
                )
            )

        return DoubleEntryJournalEntry(
            case_id=decision.case_id,
            decision_id=decision.decision_id,
            lines=lines
        )
