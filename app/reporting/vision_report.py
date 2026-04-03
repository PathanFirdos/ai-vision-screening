from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from datetime import datetime


class VisionReportGenerator:

    def generate(self, results, filename="vision_screening_report.pdf"):

        styles = getSampleStyleSheet()

        elements = []

        title = Paragraph("AI Vision Screening Report", styles['Title'])
        elements.append(title)

        elements.append(Spacer(1, 20))

        date = Paragraph(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        )

        elements.append(date)

        elements.append(Spacer(1, 20))

        data = [
            ["Metric", "Result"],
            ["Face Detected", str(results["face_detected"])],
            ["Eyes Detected", str(results["eyes_detected"])],
            ["Pupils Detected", str(results["pupils_detected"])],
            ["Gaze Direction", results["gaze_direction"]],
            ["Eye Alignment", results["alignment"]],
            ["Distance From Camera (cm)", str(results.get("distance_cm"))],
        ]

        table = Table(data, colWidths=[8 * cm, 8 * cm])

        elements.append(table)

        doc = SimpleDocTemplate(filename, pagesize=A4)

        doc.build(elements)

        return filename