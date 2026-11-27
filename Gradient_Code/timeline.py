"""Exact timeline and cue timestamps synchronized to Krisha Jogiya's voice-over."""

FPS = 30
TOTAL_DURATION = 292.0  # 4m 52s (recommended ~5 min, strict max 7 min)
AUDIO_DURATION = 283.41

STUDENT_NAME = "Krisha Jogiya"
STUDENT_ID = "E25B070935"
COURSE_TITLE = "Linear Algebra & Calculus (M3) • Internal Assessment"
VIDEO_TITLE = "THE GRADIENT: Understanding the Direction of Steepest Ascent"

SCENES = [
    ("opening", 0.00, 8.80, "OPENING", "Title & Introduction"),
    ("scene1", 8.80, 42.10, "SCENE 01 / 06", "Hook: The Mountain Problem"),
    ("scene2", 42.10, 96.80, "SCENE 02 / 06", "Concept & Partial Derivatives"),
    ("scene3", 96.80, 146.50, "SCENE 03 / 06", "Visual Intuition & Contours"),
    ("scene4", 146.50, 221.80, "SCENE 04 / 06", "Worked Example: f(x, y) = x² + y²"),
    ("scene5", 221.80, 256.30, "SCENE 05 / 06", "Applications: AI & Physics"),
    ("scene6", 256.30, 281.20, "SCENE 06 / 06", "Summary & Key Takeaways"),
    ("closing", 281.20, TOTAL_DURATION, "CLOSING", "Summary, References & Credits"),
]


def get_scene_info(t: float):
    """Return (scene_id, start, end, badge, subtitle) for global time t."""
    for s in SCENES:
        if s[1] <= t < s[2]:
            return s
    return SCENES[-1]
