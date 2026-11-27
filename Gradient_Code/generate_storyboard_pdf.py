"""Generate the 6-Scene Storyboard PDF required for M3 Internal Assessment."""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
OUT_PDF = os.path.join(REPO_DIR, "Gradient_Storyboard.pdf")


def build_storyboard_pdf():
    doc = SimpleDocTemplate(
        OUT_PDF,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Gradient: 6-Scene Storyboard - Krisha Jogiya",
        author="Krisha Jogiya (E25B070935)",
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0F172A")
    cyan = colors.HexColor("#0284C7")
    slate = colors.HexColor("#475569")
    soft_bg = colors.HexColor("#F1F5F9")
    border_col = colors.HexColor("#CBD5E1")

    s_kicker = ParagraphStyle("Kicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, textColor=cyan, spaceAfter=2)
    s_title = ParagraphStyle("Title", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=navy, spaceAfter=4)
    s_meta = ParagraphStyle("Meta", parent=styles["Normal"], fontName="Helvetica", fontSize=10, textColor=slate, spaceAfter=8)
    s_h2 = ParagraphStyle("H2", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=navy, spaceBefore=6, spaceAfter=4)
    s_obj = ParagraphStyle("Obj", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=10.5, leading=14, textColor=navy)
    s_cell_h = ParagraphStyle("CellH", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9.5, leading=12, textColor=colors.white)
    s_cell_b = ParagraphStyle("CellB", parent=styles["Normal"], fontName="Helvetica", fontSize=8.8, leading=12, textColor=navy)
    s_cell_t = ParagraphStyle("CellT", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9.0, leading=12, textColor=navy)
    s_foot = ParagraphStyle("Foot", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=slate)

    story = []
    story.append(Paragraph("LINEAR ALGEBRA AND CALCULUS (M3) • INTERNAL ASSESSMENT (10%)", s_kicker))
    story.append(Paragraph("6-Scene Storyboard: The Gradient & Steepest Ascent", s_title))
    story.append(Paragraph(
        "<b>Student:</b> Krisha Jogiya &nbsp;&nbsp;|&nbsp;&nbsp; <b>Student ID:</b> E25B070935 &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Topic:</b> Gradient (Multivariable Calculus) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Total Runtime:</b> 4m 52s (1080p MP4)",
        s_meta,
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=cyan, spaceAfter=8))

    # One-sentence learning objective box (explicitly requested in rubric Page 3)
    obj_table = Table(
        [[Paragraph(
            "<b>One-Sentence Learning Objective:</b> By the end of this video, the viewer will be able to define the gradient vector "
            "using partial derivatives, explain geometrically why it is perpendicular to contour lines and points in the direction "
            "of steepest ascent, and compute its unit direction vector for a multivariable surface.",
            s_obj,
        )]],
        colWidths=[178 * mm],
    )
    obj_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft_bg),
        ("BOX", (0, 0), (-1, -1), 0.8, cyan),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(obj_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Scene-by-Scene Visual, Narration & Mathematical Plan", s_h2))

    rows = [
        [
            Paragraph("Scene &amp; Sync", s_cell_h),
            Paragraph("Purpose", s_cell_h),
            Paragraph("Visual &amp; Animation Plan", s_cell_h),
            Paragraph("Narration &amp; Math Content", s_cell_h),
        ],
        [
            Paragraph("<b>Opening Frame</b><br/>0:00 – 0:08.8", s_cell_t),
            Paragraph("Identify course, concept, and presenter.", s_cell_b),
            Paragraph("Dark slate title card with animated concentric contours, gradient emblem, student name &amp; ID.", s_cell_b),
            Paragraph("Krisha introduces topic: <i>Understanding the Direction of Steepest Ascent</i>.", s_cell_b),
        ],
        [
            Paragraph("<b>1. Hook / Context</b><br/>0:08.8 – 0:42.1", s_cell_t),
            Paragraph("Spark curiosity via mountain climb problem.", s_cell_b),
            Paragraph("3D mountain surface with hiker point P, animated uphill (green), downhill (rose), and level (violet) paths; compass sweep + live slope bar locking onto cyan gradient arrow.", s_cell_b),
            Paragraph("Standing on a mountain, different paths give different height changes. How do we find the fastest climb? Answer: the <b>Gradient</b>.", s_cell_b),
        ],
        [
            Paragraph("<b>2. Concept Intro</b><br/>0:42.1 – 1:36.8", s_cell_t),
            Paragraph("Define notation, surface height, and partial derivatives.", s_cell_b),
            Paragraph("Transition from 1D curve y = g(x) to 3D surface z = f(x, y); plumb line from (x, y) to height f(x, y); x-slice (amber) and y-slice (emerald) curves; vector sum parallelogram.", s_cell_b),
            Paragraph("For two variables f(x, y), <b>∇f(x, y) = (∂f/∂x, ∂f/∂y)</b>. First component measures rate in x; second measures rate in y; combined vector gives steepest ascent.", s_cell_b),
        ],
        [
            Paragraph("<b>3. Visual Intuition</b><br/>1:36.8 – 2:26.5", s_cell_t),
            Paragraph("Prove why ∇f gives max rate &amp; is ⊥ to contours.", s_cell_b),
            Paragraph("360° rotating unit direction û(θ) on 3D hillside paired with live cosine dial D<sub>û</sub>f = ||∇f|| cos θ peaking at θ = 0°; camera tilts top-down to 2D contours with 90° square &amp; sliding bead.", s_cell_b),
            Paragraph("Rate of change depends on direction; max increase occurs along ∇f (cos 0° = 1). Along contour lines Δf = 0, so ∇f is perpendicular to level curves.", s_cell_b),
        ],
        [
            Paragraph("<b>4. Worked Example &amp; Interpretation</b><br/>2:26.5 – 3:41.8", s_cell_t),
            Paragraph("Step-by-step calculation and geometric meaning.", s_cell_b),
            Paragraph("Polar 3D paraboloid bowl f(x, y) = x² + y² revealing point (1, 1, 2) and arrow (2, 2); transitions to 2D circular contours showing step across f = 2→4→6 and shrinking (2, 2) to unit circle.", s_cell_b),
            Paragraph("f(x, y) = x² + y² ⇒ ∂f/∂x = 2x, ∂f/∂y = 2y ⇒ ∇f = (2x, 2y). At (1, 1): ∇f(1, 1) = (2, 2). Magnitude ||(2, 2)|| = 2√2 ⇒ unit direction û = (1/√2, 1/√2).", s_cell_b),
        ],
        [
            Paragraph("<b>5. Applications</b><br/>3:41.8 – 4:16.3", s_cell_t),
            Paragraph("Connect gradient to ML and Physics.", s_cell_b),
            Paragraph("Dual cards: (1) ML Gradient Descent trajectory stepping along −∇L down loss contours with live loss readout; (2) 2D thermal heatmap with ∇T vector field pointing toward hot source.", s_cell_b),
            Paragraph("Machine learning uses −∇L (gradient descent) to minimize errors; physics uses ∇T and ∇P to model temperature and pressure change across space.", s_cell_b),
        ],
        [
            Paragraph("<b>6. Summary &amp; Closing Credits</b><br/>4:16.3 – 4:52.0", s_cell_t),
            Paragraph("Reinforce 3 takeaways and disclose sources/AI.", s_cell_b),
            Paragraph("Three takeaway cards + stylized mountain silhouette climber ascending along ∇f; Closing Frame with references, music credit, and AI disclosure.", s_cell_b),
            Paragraph("Recap: (1) vector of partial derivatives, (2) steepest increase direction, (3) perpendicular to contours. Mountain recall &amp; thank you.", s_cell_b),
        ],
    ]

    table = Table(rows, colWidths=[31 * mm, 30 * mm, 61 * mm, 56 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, soft_bg]),
        ("GRID", (0, 0), (-1, -1), 0.5, border_col),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Verification, Credits &amp; Originality Disclosure:</b> All mathematical derivations (f(x, y) = x² + y², partial derivatives, "
        "gradient evaluation at (1, 1), magnitude 2√2, and unit vector (1/√2, 1/√2)) were verified by hand. Voice-over narration was "
        "recorded by Krisha Jogiya. Background ambient pad was synthesized procedurally in Python (NumPy/SciPy). AI assistance "
        "(Antigravity / Gemini) was used for storyboard structuring, timestamp alignment, and Python visualization code.",
        s_foot,
    ))

    doc.build(story)
    print("Generated Storyboard PDF:", OUT_PDF)


if __name__ == "__main__":
    build_storyboard_pdf()
