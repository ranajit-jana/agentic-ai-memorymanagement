"""Run once to generate sample_data/support_policy.pdf"""
from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

sections = [
    (
        "YNC E-Commerce Customer Support Policy",
        None,
    ),
    (
        "1. Refund Policy",
        (
            "Customers are eligible for a full refund within 30 days of purchase, provided the "
            "product is unused and in its original packaging. Refunds are processed within 5-7 "
            "business days after the returned item is received and inspected. Digital products and "
            "downloadable content are non-refundable once accessed. Shipping costs are "
            "non-refundable unless the return is due to a defective or incorrect item."
        ),
    ),
    (
        "2. Shipping & Delivery Policy",
        (
            "Standard shipping takes 3-7 business days. Express shipping (1-2 business days) is "
            "available at additional cost. YNC is not responsible for delays caused by courier "
            "services or customs. If an order has not arrived within 10 business days of the "
            "estimated delivery date, customers should contact support with their order number. "
            "A replacement or refund will be issued for confirmed lost shipments."
        ),
    ),
    (
        "3. Product Defect & Replacement Policy",
        (
            "Products found to be defective on arrival must be reported within 7 days of delivery. "
            "Customers are required to provide photographic evidence of the defect. Upon "
            "verification, YNC will ship a replacement unit at no additional cost. If the exact "
            "product is unavailable, a full refund will be issued. Defective items must be returned "
            "via a prepaid return label provided by YNC support."
        ),
    ),
    (
        "4. Account & Billing Policy",
        (
            "Customers experiencing account access issues should use the self-service password "
            "reset tool. Support agents may assist with account unlocking after identity "
            "verification. Billing disputes must be raised within 60 days of the charge date. "
            "Duplicate charges will be investigated and refunded within 5 business days. "
            "YNC does not store full credit card details; all transactions are processed through "
            "a PCI-DSS compliant payment gateway."
        ),
    ),
    (
        "5. Escalation & SLA Policy",
        (
            "All support tickets are acknowledged within 1 hour. Standard tickets are resolved "
            "within 24 hours. High-priority tickets (billing, defective products) are resolved "
            "within 4 hours. Critical tickets (data breach, mass order failure) are escalated "
            "immediately to the senior support team and resolved within 2 hours. SLA breaches "
            "are reviewed weekly by the support manager."
        ),
    ),
    (
        "6. Privacy & Data Policy",
        (
            "YNC complies with GDPR and applicable data protection laws. Customer personal data "
            "is used solely for order processing and support purposes. Customers may request "
            "access to, correction of, or deletion of their personal data by contacting "
            "privacy@ync.com. Data deletion requests are processed within 30 days. YNC does not "
            "sell customer data to third parties."
        ),
    ),
    (
        "7. Discount & Promotional Code Policy",
        (
            "Promotional codes must be applied at checkout and cannot be applied retroactively "
            "to placed orders. In cases where a valid code failed to apply due to a system error, "
            "support agents may issue an equivalent account credit. Codes are single-use unless "
            "stated otherwise. Expired codes cannot be honoured."
        ),
    ),
]

for title, body in sections:
    pdf.add_page()
    if body is None:
        pdf.set_font("Helvetica", "B", 20)
        pdf.ln(60)
        pdf.multi_cell(0, 12, title, align="C")
    else:
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 10, title)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 8, body)

pdf.output("sample_data/support_policy.pdf")
print("Created sample_data/support_policy.pdf")
