"""Canonical Financial Data Models, Enums, and Provenance Schemas for Prototype 3."""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timezone
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class SourceSystem(str, Enum):
    BANK = "BANK"
    RAZORPAY = "RAZORPAY"
    GATEWAY = "GATEWAY"
    ERP = "ERP"
    GST = "GST"
    SETTLEMENT = "SETTLEMENT"
    MERCHANT_PORTAL = "MERCHANT_PORTAL"
    CRM = "CRM"


class TransactionType(str, Enum):
    PAYMENT = "PAYMENT"
    REFUND = "REFUND"
    CHARGEBACK = "CHARGEBACK"
    REVERSAL = "REVERSAL"
    TRANSFER = "TRANSFER"
    FEE = "FEE"
    PAYOUT = "PAYOUT"
    INVOICE = "INVOICE"


class PaymentMethod(str, Enum):
    UPI = "UPI"
    CARD = "CARD"
    NETBANKING = "NETBANKING"
    WALLET = "WALLET"
    EMI = "EMI"
    NA = "NA"


class TransactionStatus(str, Enum):
    CAPTURED = "CAPTURED"
    REFUNDED = "REFUNDED"
    FAILED = "FAILED"
    REVERSED = "REVERSED"
    PENDING = "PENDING"


class DecisionLabel(str, Enum):
    MATCHED = "MATCHED"
    EXCEPTION = "EXCEPTION"
    UNCERTAIN = "UNCERTAIN"


class ReasonCode(str, Enum):
    # Match Reasons
    EXACT_IDENTIFIER_MATCH = "EXACT_IDENTIFIER_MATCH"
    FEE_ADJUSTED_MATCH = "FEE_ADJUSTED_MATCH"
    GST_ADJUSTED_MATCH = "GST_ADJUSTED_MATCH"
    SPLIT_PAYMENT_MATCH = "SPLIT_PAYMENT_MATCH"
    TIMING_ALIGNED_MATCH = "TIMING_ALIGNED_MATCH"
    FUZZY_ENTITY_MATCH = "FUZZY_ENTITY_MATCH"
    GRAPH_RESOLVED_MATCH = "GRAPH_RESOLVED_MATCH"
    REVERSAL_MATCH = "REVERSAL_MATCH"

    # Exception Reasons
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    DUPLICATE_TRANSACTION = "DUPLICATE_TRANSACTION"
    DUPLICATE_REVERSAL = "DUPLICATE_REVERSAL"
    EXPIRED_REVERSAL = "EXPIRED_REVERSAL"
    GST_CALCULATION_ERROR = "GST_CALCULATION_ERROR"
    INVALID_REVERSAL = "INVALID_REVERSAL"
    ORPHAN_REFUND = "ORPHAN_REFUND"
    UNAUTHORIZED_CHARGE = "UNAUTHORIZED_CHARGE"
    MISSING_AUTHORIZATION = "MISSING_AUTHORIZATION"
    SETTLEMENT_OVERPAYMENT = "SETTLEMENT_OVERPAYMENT"
    DATE_OUT_OF_BOUNDS = "DATE_OUT_OF_BOUNDS"
    REVENUE_LEAKAGE_DETECTED = "REVENUE_LEAKAGE_DETECTED"
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"

    # Uncertain / Abstention Reasons
    AMBIGUOUS_CANDIDATES = "AMBIGUOUS_CANDIDATES"
    MISSING_SOURCE_RECORD = "MISSING_SOURCE_RECORD"
    INSUFFICIENT_PROVENANCE = "INSUFFICIENT_PROVENANCE"
    INVESTIGATION_BUDGET_EXHAUSTED = "INVESTIGATION_BUDGET_EXHAUSTED"
    UNSEEN_TRANSACTION_STRUCTURE = "UNSEEN_TRANSACTION_STRUCTURE"
    BELOW_CONFIDENCE_THRESHOLD = "BELOW_CONFIDENCE_THRESHOLD"


class FieldProvenance(BaseModel):
    """Tracks field-level transformation history from raw input to canonical representation."""
    model_config = ConfigDict(frozen=True)

    source_system: SourceSystem
    source_field_name: str
    raw_value: str
    normalized_value: str
    transformation_applied: str
    transformation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_score: float = 1.0


class RawSourceRecord(BaseModel):
    """Preserves immutable raw input exactly as received from external source."""
    model_config = ConfigDict(frozen=True)

    raw_record_id: str
    source_system: SourceSystem
    source_file_or_endpoint: str
    raw_payload: Dict[str, Any]
    ingestion_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_hash: str


class CanonicalTransaction(BaseModel):
    """
    Normalized, canonical representation of a financial transaction with full lineage.
    Monetary values stored as exact integer paise (1 INR = 100 paise) and synced with Decimal amounts.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transaction_id: str
    source_system: SourceSystem
    source_record_id: str
    ground_truth_tx_id: Optional[str] = None  # Hidden latent ID for benchmarking

    # Monetary fields (Integer Paise & Decimal)
    amount_paise: int = 0
    fee_paise: int = 0
    gst_paise: int = 0
    net_paise: int = 0

    amount: Decimal = Decimal("0.00")
    currency: str = "INR"
    fee: Decimal = Decimal("0.00")
    tax: Decimal = Decimal("0.00")
    net_amount: Decimal = Decimal("0.00")

    # Temporal fields (Epoch UTC seconds & datetime)
    txn_timestamp: int = 0
    settlement_timestamp_epoch: Optional[int] = None
    transaction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    settlement_timestamp: Optional[datetime] = None

    # Entities and References
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    merchant_id: Optional[str] = None
    merchant_name: Optional[str] = None
    merchant_name_norm: Optional[str] = None
    counterparty_name: Optional[str] = None
    account_number_mask: Optional[str] = None

    # Identifiers & Descriptions
    order_id: Optional[str] = None
    invoice_reference: Optional[str] = None
    payment_reference: Optional[str] = None
    batch_reference: Optional[str] = None
    utr: Optional[str] = None
    vpa: Optional[str] = None
    gstin: Optional[str] = None
    approval_code: Optional[str] = None
    reason_code: Optional[str] = None

    raw_narrative: str = ""
    normalized_narrative: str = ""

    # Transaction State
    payment_method: PaymentMethod = PaymentMethod.NA
    status: TransactionStatus = TransactionStatus.CAPTURED
    transaction_type: TransactionType = TransactionType.PAYMENT
    is_reversal: bool = False
    is_refund: bool = False
    is_split: bool = False
    parent_transaction_id: Optional[str] = None

    # Field-level lineage
    provenance: Dict[str, FieldProvenance] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        # Sync amount_paise and amount if either is zero and the other is set
        if self.amount_paise == 0 and self.amount != Decimal("0.00"):
            self.amount_paise = int(round(self.amount * 100))
        elif self.amount == Decimal("0.00") and self.amount_paise != 0:
            self.amount = Decimal(self.amount_paise) / Decimal(100)

        if self.fee_paise == 0 and self.fee != Decimal("0.00"):
            self.fee_paise = int(round(self.fee * 100))
        elif self.fee == Decimal("0.00") and self.fee_paise != 0:
            self.fee = Decimal(self.fee_paise) / Decimal(100)

        if self.gst_paise == 0 and self.tax != Decimal("0.00"):
            self.gst_paise = int(round(self.tax * 100))
        elif self.tax == Decimal("0.00") and self.gst_paise != 0:
            self.tax = Decimal(self.gst_paise) / Decimal(100)

        if self.net_paise == 0:
            self.net_paise = self.amount_paise - self.fee_paise - self.gst_paise
            self.net_amount = Decimal(self.net_paise) / Decimal(100)

        if self.txn_timestamp == 0 and self.transaction_timestamp:
            self.txn_timestamp = int(self.transaction_timestamp.timestamp())
        elif self.txn_timestamp != 0 and (not self.transaction_timestamp or self.transaction_timestamp.year == 1970):
            self.transaction_timestamp = datetime.fromtimestamp(self.txn_timestamp, tz=timezone.utc)

        if self.settlement_timestamp_epoch and not self.settlement_timestamp:
            self.settlement_timestamp = datetime.fromtimestamp(self.settlement_timestamp_epoch, tz=timezone.utc)
        elif self.settlement_timestamp and not self.settlement_timestamp_epoch:
            self.settlement_timestamp_epoch = int(self.settlement_timestamp.timestamp())

        if not self.merchant_name_norm and self.merchant_name:
            self.merchant_name_norm = self.merchant_name.lower().strip()

    @field_validator("amount", "fee", "tax", "net_amount", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))


class PartyEntity(BaseModel):
    """Canonical party (Customer or Merchant) master record."""
    entity_id: str
    entity_type: str  # "CUSTOMER" | "MERCHANT"
    primary_name: str
    aliases: List[str] = Field(default_factory=list)
    tax_identifier: Optional[str] = None
    gstin: Optional[str] = None
    email_domain: Optional[str] = None
    trusted_source: bool = False
    historical_reconciliation_rate: float = 0.95
    risk_tier: str = "LOW"  # LOW | MEDIUM | HIGH


class InvoiceRecord(BaseModel):
    """Canonical invoice record from ERP or Billing system."""
    invoice_id: str
    invoice_number: str
    customer_id: str
    issue_date: date
    due_date: date
    subtotal: Decimal
    tax_amount: Decimal
    total_amount: Decimal
    paid_amount: Decimal = Decimal("0.00")
    status: str = "ISSUED"  # ISSUED, PAID, PARTIALLY_PAID, CANCELLED
    currency: str = "INR"
    ground_truth_tx_id: Optional[str] = None


class SettlementBatch(BaseModel):
    """Canonical batch settlement payout from payment processor / merchant acquirer."""
    batch_id: str
    processor_name: str
    payout_date: date
    gross_volume: Decimal
    total_fees: Decimal
    net_payout: Decimal
    currency: str = "INR"
    transaction_count: int
    included_transaction_ids: List[str] = Field(default_factory=list)
    ground_truth_batch_id: Optional[str] = None


class CandidateMatch(BaseModel):
    """A proposed relationship between two or more canonical transactions."""
    source_record: CanonicalTransaction
    candidate_records: List[CanonicalTransaction]
    match_type: str = "1:1"  # "1:1", "1:N", "N:1"
    
    # Feature scores
    identifier_score: float = 0.0
    amount_score: float = 0.0
    date_score: float = 0.0
    lexical_score: float = 0.0
    embedding_score: float = 0.0
    graph_score: float = 0.0
    rule_score: float = 0.0
    composite_similarity: float = 0.0
    
    blocking_keys_matched: List[str] = Field(default_factory=list)
    retrieval_stage: str = "EXACT_BLOCKING"
    notes: List[str] = Field(default_factory=list)


class AgentRecommendation(BaseModel):
    """Structured output emitted by the Gemini Vertex AI Investigation Agent."""
    case_id: str
    recommended_decision: DecisionLabel
    primary_reason: ReasonCode
    cited_evidence_ids: List[str] = Field(default_factory=list)
    matched_record_ids: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    fuzzy_score: float = 0.0
    leakage_risk: float = 0.0
    rules_passed: List[str] = Field(default_factory=list)
    rules_failed: List[str] = Field(default_factory=list)
    rules_warned: List[str] = Field(default_factory=list)
    explanation_narrative: str = ""
    tool_calls_performed: int = 0
    tool_call_sequence: List[str] = Field(default_factory=list)
    investigation_hypotheses_tested: List[str] = Field(default_factory=list)
    human_review_required: bool = False
    investigator: str = "deterministic-fast-path"


class FinalDecisionRecord(BaseModel):
    """The authoritative, audited decision produced by the Deterministic Policy Verifier."""
    decision_id: str
    case_id: str
    decision: DecisionLabel
    reason: ReasonCode
    calibrated_confidence: float
    is_automated: bool
    requires_human_review: bool

    matched_pairs: List[Dict[str, str]] = Field(default_factory=list)
    source_record_ids: List[str] = Field(default_factory=list)
    cited_evidence_ids: List[str] = Field(default_factory=list)
    
    rules_passed: List[str] = Field(default_factory=list)
    rules_failed: List[str] = Field(default_factory=list)
    rules_warned: List[str] = Field(default_factory=list)
    leakage_risk: float = 0.0
    tool_calls_count: int = 0
    
    verifier_status: str = "VERIFIED_VALID"
    verifier_notes: List[str] = Field(default_factory=list)
    explanation: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Human Review Annotation if applicable
    human_reviewer_id: Optional[str] = None
    human_final_label: Optional[DecisionLabel] = None
    human_notes: Optional[str] = None
    human_review_timestamp: Optional[datetime] = None
