# The Gradient: Understanding the Direction of Steepest Ascent

**Linear Algebra and Calculus (M3) • Internal Assessment (Project Option 1: Animation Video)**  
**Student:** Krisha Jogiya  
**Student ID:** `E25B070935`  
**Topic:** Gradient (`Multivariable Calculus`)

---

## Repository Contents

- **`Gradient_Video.mp4`** — Final 1080p (1920×1080, 16:9, 30 fps) rendered animation video synced to voice-over narration (`4m 52s`), including Opening Frame and Closing Summary/Credits Frame.
- **`Gradient_Storyboard.pdf`** — 6-Scene Storyboard document with the learning objective, scene breakdown, visual/math plan, and synced timestamps.
- **`Gradient_Code/`** — Complete, runnable Python animation source code and audio assets:
  - `build_video.py` — Parallel frame renderer and FFmpeg video/audio muxer
  - `timeline.py` — Exact scene boundaries and cue timestamps synced to the recorded voice-over
  - `styles.py` — Design tokens, color coding, easing curves, and Matplotlib STIX math renderer
  - `surface3d.py` — Custom right-handed 3D camera projection and back-to-front painter's renderer
  - `audio_mix.py` — Voice-over cleanup/loudness normalization and subtle ducked ambient underscore
  - `generate_storyboard_pdf.py` — Script that builds `Gradient_Storyboard.pdf`
  - `scenes/` — Modular scene scripts (`opening_closing.py`, `scene1_hook.py` … `scene6_summary.py`)
  - `assets/voiceover.m4a` — Recorded narration audio

---

## How to Reproduce

Requires Python 3.10+ and `ffmpeg` on `PATH`:

```bash
cd Gradient_Code
pip install -r requirements.txt

# Render representative preview keyframes for every scene
python3 build_video.py --preview

# Render the full 1080p MP4 synced with voice-over
python3 build_video.py --workers 8

# Regenerate the 6-Scene Storyboard PDF
python3 generate_storyboard_pdf.py
```

---

## Quick Mathematical & Code Walkthrough (for Viva / Q&A)

1. **Definition (`scenes/scene2_definition.py`):**  
   For a differentiable two-variable function $z = f(x, y)$, the gradient is the vector of first-order partial derivatives:
   $$\nabla f(x, y) = \left(\frac{\partial f}{\partial x}, \; \frac{\partial f}{\partial y}\right)$$
2. **Why Steepest Ascent & Contour Perpendicularity (`scenes/scene3_visual.py`):**  
   The directional derivative along any unit vector $\hat{u}$ making angle $\theta$ with $\nabla f$ is:
   $$D_{\hat{u}}f = \nabla f \cdot \hat{u} = \|\nabla f\|\cos\theta$$
   - Maximum rate $+\|\nabla f\|$ occurs at $\theta = 0^\circ$ ($\hat{u} \parallel \nabla f$).
   - Along a contour line $f(x, y) = c$, height does not change ($\Delta f = 0$), so $\nabla f \cdot \hat{t} = 0 \implies \nabla f \perp \hat{t}$.
3. **Worked Example (`scenes/scene4_example.py`):**  
   - Function: $f(x, y) = x^2 + y^2$ (polar paraboloid mesh in `surface3d.py`).
   - Partials: $\frac{\partial f}{\partial x} = 2x$, $\frac{\partial f}{\partial y} = 2y \implies \nabla f(x, y) = (2x, 2y)$.
   - At $(1, 1)$: $\nabla f(1, 1) = (2, 2)$ with magnitude $\|\nabla f(1, 1)\| = \sqrt{2^2 + 2^2} = 2\sqrt{2}$.
   - Unit steepest ascent vector: $\hat{u} = \frac{(2, 2)}{2\sqrt{2}} = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$.
4. **Applications (`scenes/scene5_apps.py`):**  
   - **Machine Learning:** Gradient descent updates weights via $w_{\mathrm{new}} = w - \eta \nabla L(w)$ (moving opposite $+\nabla L$ to minimize error).
   - **Physics:** Spatial gradients $\nabla T$ and $\nabla P$ point in the direction of fastest increase across space.

---

## Credits & AI Disclosure

- **Script & Voice-Over:** Krisha Jogiya (`E25B070935`)
- **Mathematical References:** James Stewart (*Calculus: Early Transcendentals*); 3Blue1Brown & Khan Academy intuition on directional derivatives and gradients.
- **Background Audio:** Subtle procedural ambient harmonic pad synthesized in Python (`NumPy`/`SciPy`) via `audio_mix.py`.
- **AI Disclosure:** AI assistance (Antigravity / Gemini) was used to assist with storyboard structuring, audio-to-visual timestamp alignment, and Python visualization code (`NumPy`, `Matplotlib`, `Pillow`, `FFmpeg`).
