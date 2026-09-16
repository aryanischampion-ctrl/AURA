from core.detector import detect_sensitive_data
from core.ai_analyzer import analyze_context
from core.risk_engine import calculate_risk
from core.recommender import generate_recommendation
from core.sanitizer import sanitize_text
import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QFrame,
    QComboBox,
    QScrollArea,
)

from PySide6.QtCore import Qt


class AuraApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AURA | Adaptive User Risk Awareness")
        self.setMinimumSize(800, 600)

        self.setup_ui()

    def setup_ui(self):

        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        # Title
        title = QLabel("AURA")
        title.setObjectName("title")

        subtitle = QLabel(
            "Adaptive User Risk Awareness"
        )
        subtitle.setObjectName("subtitle")
        ai_status = QLabel(
            "AI ENGINE  •  INT8 ONNX  •  482 Features  •  Local Processing"
        )
        ai_status.setObjectName("ai_status")

        description = QLabel(
            "Analyze text for potentially sensitive information "
            "before sharing it with external services."
        )
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(ai_status)


        # Input label
        input_label = QLabel("Enter text to analyze")
        input_label.setObjectName("section_label")

        layout.addWidget(input_label)
        destination_label = QLabel("Where are you about to share this?")
        destination_label.setObjectName("section_label")
        layout.addWidget(destination_label)

        self.destination = QComboBox()
        self.destination.addItems([
            "External AI Tool",
            "Public Website",
            "Email",
            "Cloud Storage",
            "Private / Local"
        ])
        layout.addWidget(self.destination)

        # Text input
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText(
            "Example: OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx"
        )

        self.text_input.setMinimumHeight(150)

        layout.addWidget(self.text_input)

        # Analyze button
        self.analyze_button = QPushButton("Analyze")

        self.analyze_button.clicked.connect(
            self.analyze_text
        )

        layout.addWidget(self.analyze_button)
        self.safe_button = QPushButton("Make Content Safe")
        self.safe_button.clicked.connect(self.make_content_safe)
        layout.addWidget(self.safe_button)
        # Result card
        self.result_card = QFrame()
        self.result_card.setObjectName("result_card")

        result_layout = QVBoxLayout()

        self.result_title = QLabel("Analysis Result")
        self.result_title.setObjectName("result_title")

        self.result_label = QLabel(
            "Enter text and click Analyze."
        )
        self.result_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        result_layout.addWidget(self.result_title)
        result_layout.addWidget(self.result_label)

        self.result_card.setLayout(result_layout)

        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setWidget(self.result_card)
        result_scroll.setMinimumHeight(180)

        layout.addWidget(result_scroll)

        # Footer
        footer = QLabel(
            "AURA Prototype • Local analysis"
        )

        footer.setAlignment(Qt.AlignCenter)

        layout.addWidget(footer)

        self.setLayout(layout)

        self.apply_styles()

    def analyze_text(self):

        text = self.text_input.toPlainText().strip()

        if not text:

            self.result_label.setText(
                "Please enter some text first."
            )

            return
        findings = detect_sensitive_data(text)

        destination = self.destination.currentText()
        ai_result = analyze_context(text)
        risk = calculate_risk(
            findings,
            destination,
            ai_result
        )

        recommendation = generate_recommendation(
            findings,
            risk["level"],
            destination,
        )

        # Step 4: Prepare result
        result = (
            f"🛡️ AURA SECURITY ANALYSIS\n\n"
            f"Risk Level: {risk['level']}\n"
            f"Risk Score: {risk['score']}/100\n\n"
            f"Destination:\n{destination}\n\n"
            f"AI Understanding:\n"
            f"{ai_result['intent']}\n"
            f"Confidence: {ai_result['confidence']}%\n\n"
        )

        if findings:
            result += "Detected Information:\n"

            for finding in findings:
                result += (
                    f"• {finding['type']} "
                    f"({finding['severity'].upper()})\n"
                )
        else:
            result += "No sensitive information detected.\n"

        result += f"\nRecommendation:\n{recommendation}"

        self.result_label.setText(result)

    def make_content_safe(self):
        text = self.text_input.toPlainText().strip()

        if not text:
            self.result_label.setText("Please enter some text first.")
            return

        safe_text = sanitize_text(text)

        self.result_label.setText(
            "🛡️ SAFE VERSION\n\n"
            + safe_text
            + "\n\nSensitive information has been redacted."
        )

    def apply_styles(self):

        self.setStyleSheet("""

            QWidget {
                background-color: #f5f7fb;
                color: #172033;
                font-family: Arial;
                font-size: 14px;
            }

            QLabel#title {
                font-size: 32px;
                font-weight: bold;
                color: #172033;
            }

            QLabel#subtitle {
                font-size: 16px;
                color: #5d6b82;
            }
            QLabel#ai_status {
                font-size: 12px;
                font-weight: bold;
                color: #315efb;
                padding: 6px 0px;
            }

            QLabel#section_label {
                font-size: 15px;
                font-weight: bold;
            }

            QTextEdit {
                background-color: white;
                border: 1px solid #d8deea;
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
            }

            QPushButton {
                background-color: #315efb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #244bd1;
            }

            QFrame#result_card {
                background-color: white;
                border: 1px solid #d8deea;
                border-radius: 10px;
                padding: 15px;
            }

            QLabel#result_title {
                font-size: 17px;
                font-weight: bold;
            }

        """)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = AuraApp()
    window.show()

    sys.exit(app.exec())