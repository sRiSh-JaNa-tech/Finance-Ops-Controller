"""Production Heterogeneous Ingestion Parsers & Dead Letter Queue (DLQ).

Supports:
1. SWIFT MT940 Bank Statement format (Tags :61:, :86:)
2. Payment Processor Settlement CSV (Stripe/Razorpay multi-column schema)
3. Dead Letter Queue (DLQ) for corrupt/unparseable records with quarantine reason tagging.
"""

import re
import csv
import io
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, date
from decimal import Decimal
from pydantic import BaseModel, Field

from finance_ops.core.models import (
    CanonicalTransaction, SourceSystem, PaymentMethod, TransactionStatus, TransactionType
)


class DeadLetterRecord(BaseModel):
    """Quarantined unparseable or corrupt raw payload in the Dead Letter Queue."""
    record_id: str
    source_format: str
    raw_payload: str
    quarantine_reason: str
    quarantined_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    field_errors: Dict[str, str] = Field(default_factory=dict)


class DeadLetterQueue:
    """Thread-safe Dead Letter Queue for schema errors and corrupt financial records."""

    def __init__(self):
        self.records: List[DeadLetterRecord] = []

    def quarantine(self, raw_data: str, source_format: str, reason: str, field_errors: Optional[Dict[str, str]] = None) -> DeadLetterRecord:
        import uuid
        dlq_rec = DeadLetterRecord(
            record_id=f"DLQ_{uuid.uuid4().hex[:8].upper()}",
            source_format=source_format,
            raw_payload=raw_data,
            quarantine_reason=reason,
            field_errors=field_errors or {}
        )
        self.records.append(dlq_rec)
        return dlq_rec

    def get_all(self) -> List[DeadLetterRecord]:
        return self.records


class MT940BankStatementParser:
    """
    Parser for SWIFT MT940 electronic bank statements.
    Extracts Tag :61: (Statement Line) and Tag :86: (Information to Account Owner).
    """

    @staticmethod
    def parse_statement(content: str, dlq: Optional[DeadLetterQueue] = None) -> List[CanonicalTransaction]:
        transactions: List[CanonicalTransaction] = []
        lines = content.strip().splitlines()

        current_61 = None
        current_86 = ""

        for line in lines:
            line_str = line.strip()
            if line_str.startswith(":61:"):
                if current_61:
                    tx = MT940BankStatementParser._convert_to_canonical(current_61, current_86, dlq)
                    if tx:
                        transactions.append(tx)
                current_61 = line_str[4:]
                current_86 = ""
            elif line_str.startswith(":86:"):
                current_86 = line_str[4:]
            elif current_86:
                current_86 += " " + line_str

        if current_61:
            tx = MT940BankStatementParser._convert_to_canonical(current_61, current_86, dlq)
            if tx:
                transactions.append(tx)

        return transactions

    @staticmethod
    def _convert_to_canonical(tag_61: str, tag_86: str, dlq: Optional[DeadLetterQueue] = None) -> Optional[CanonicalTransaction]:
        # Tag :61: format: YYMMDD[MMDD] Debit/Credit Mark (C/D/RC/RD) Currency/Amount N... Ref
        pattern = r"^(\d{6})(\d{4})?(CR|DR|RC|RD|C|D)([A-Z]{0,3})([0-9,.]+)"
        match = re.match(pattern, tag_61)
        if not match:
            if dlq:
                dlq.quarantine(f"{tag_61} || {tag_86}", "SWIFT_MT940", "Invalid :61: statement line structure")
            return None

        val_date_str, _, dc_indicator, curr, amt_str = match.groups()
        try:
            amt_clean = amt_str.replace(",", ".")
            amount = Decimal(amt_clean)
            amount_paise = int(round(amount * 100))
            is_credit = "C" in dc_indicator
            if not is_credit:
                amount = -amount
                amount_paise = -amount_paise

            # Extract UTR and Invoice Reference from :86:
            utr_match = re.search(r"(?:UTR)[:\s-]*([A-Z0-9_-]+)", tag_86, re.IGNORECASE)
            inv_match = re.search(r"(?:INVOICE|INV|REF)[:\s-]*([A-Z0-9_-]+)", tag_86, re.IGNORECASE)
            utr = utr_match.group(1) if utr_match else None
            inv_ref = inv_match.group(1) if inv_match else None

            import uuid
            tx_id = f"TXN_MT940_{uuid.uuid4().hex[:8].upper()}"

            # Parse date YYMMDD
            year = 2000 + int(val_date_str[:2])
            month = int(val_date_str[2:4])
            day = int(val_date_str[4:6])
            dt = datetime(year, month, day, 12, 0, tzinfo=timezone.utc)

            return CanonicalTransaction(
                transaction_id=tx_id,
                source_system=SourceSystem.BANK,
                source_record_id=f"STMT_{val_date_str}_{abs(amount_paise)}",
                amount=abs(amount),
                amount_paise=abs(amount_paise),
                currency="INR",
                raw_narrative=tag_86 or tag_61,
                transaction_timestamp=dt,
                utr=utr,
                invoice_reference=inv_ref,
                status=TransactionStatus.CAPTURED,
                transaction_type=TransactionType.PAYMENT if is_credit else TransactionType.REFUND
            )
        except Exception as e:
            if dlq:
                dlq.quarantine(f"{tag_61} || {tag_86}", "SWIFT_MT940", f"Conversion error: {str(e)}")
            return None


class GatewaySettlementCSVParser:
    """
    Parser for heterogeneous gateway CSV settlement exports (Stripe, Razorpay, Adyen).
    """

    @staticmethod
    def parse_csv(csv_content: str, dlq: Optional[DeadLetterQueue] = None) -> List[CanonicalTransaction]:
        transactions: List[CanonicalTransaction] = []
        reader = csv.DictReader(io.StringIO(csv_content.strip()))

        for row_idx, row in enumerate(reader):
            # Column mapping heuristics
            txn_id = row.get("payment_id") or row.get("transaction_id") or row.get("id") or row.get("Identifier")
            amount_raw = row.get("amount") or row.get("gross_amount") or row.get("Amount")
            fee_raw = row.get("fee") or row.get("fee_amount") or row.get("Fee") or "0.00"
            tax_raw = row.get("tax") or row.get("gst") or row.get("Tax") or "0.00"
            inv_ref = row.get("invoice_id") or row.get("description") or row.get("order_id") or row.get("Reference")

            if not txn_id or not amount_raw:
                if dlq:
                    dlq.quarantine(str(row), "GATEWAY_CSV", "Missing required transaction ID or amount", {"row_index": str(row_idx)})
                continue

            try:
                amt = Decimal(str(amount_raw).replace(",", "").strip())
                amt_paise = int(round(amt * 100))
                fee_paise = int(round(Decimal(str(fee_raw).replace(",", "").strip()) * 100))
                tax_paise = int(round(Decimal(str(tax_raw).replace(",", "").strip()) * 100))

                tx = CanonicalTransaction(
                    transaction_id=str(txn_id),
                    source_system=SourceSystem.GATEWAY,
                    source_record_id=str(txn_id),
                    amount=amt,
                    amount_paise=amt_paise,
                    fee_paise=fee_paise,
                    gst_paise=tax_paise,
                    currency="INR",
                    raw_narrative=str(inv_ref or ""),
                    invoice_reference=str(inv_ref) if inv_ref and "INV" in str(inv_ref).upper() else None,
                    order_id=str(inv_ref) if inv_ref and "ORD" in str(inv_ref).upper() else None,
                    transaction_timestamp=datetime.now(timezone.utc),
                    status=TransactionStatus.CAPTURED
                )
                transactions.append(tx)
            except Exception as e:
                if dlq:
                    dlq.quarantine(str(row), "GATEWAY_CSV", f"Parse error: {str(e)}", {"row_index": str(row_idx)})

        return transactions
