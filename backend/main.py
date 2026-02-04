import os
import io
import base64
import tempfile
import logging
from typing import Optional, Literal

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO

from openai import OpenAI
from fpdf import FPDF
from dotenv import load_dotenv


load_dotenv()
# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("dentalvision")

# ----------------------------
# Config
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOLERANCE_DEG = float(os.getenv("TOLERANCE_DEG", "2.5"))
KPT_CONF_TH = float(os.getenv("KPT_CONF_TH", "0.35"))  # confidence threshold for keypoints


client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
# If you want to allow any origin during dev; lock it down for production
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="DentalVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLO("yolo11n-pose.pt")


# ----------------------------
# Schemas
# ----------------------------
class AnalyzeResponse(BaseModel):
    angle: float
    orientation: Literal["Right Profile", "Left Profile", "Unknown"]
    scale: str
    image: str  # data URL
    status: Literal["Optimal", "Needs Correction", "Unknown"]
    note: str
    tolerance_deg: float = Field(default=TOLERANCE_DEG)


class GeneratePDFRequest(BaseModel):
    angle: float
    orientation: str
    scale: str
    status: str
    note: str
    image: str  # data URL base64


# ----------------------------
# Helpers
# ----------------------------
def _validate_image_upload(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png).")


def _decode_image(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Try JPG/PNG.")
    return img


def _angle_between(p1, p2) -> float:
    """Angle in degrees of line p1->p2 relative to horizontal axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    raw = np.degrees(np.arctan2(dy, dx))

    # Normalize to [-90, 90] for easier interpretation
    if raw > 90:
        raw -= 180
    elif raw < -90:
        raw += 180
    return float(np.round(raw, 2))


def _status_from_angle(angle: float) -> str:
    if np.isnan(angle):
        return "Unknown"
    return "Optimal" if abs(angle) <= TOLERANCE_DEG else "Needs Correction"


def _dataurl_from_bgr(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _extract_keypoints(results) -> Optional[tuple]:
    """
    Returns (xy, conf) arrays for the first person if available.
    ultralytics pose:
      xy: (N, K, 2), conf: (N, K)
    """
    if results.keypoints is None:
        return None
    xy = results.keypoints.xy
    conf = getattr(results.keypoints, "conf", None)
    if xy is None or len(xy) == 0:
        return None
    xy0 = xy[0].cpu().numpy()
    conf0 = conf[0].cpu().numpy() if conf is not None else None
    return xy0, conf0


def _pick_profile_points(xy: np.ndarray, conf: Optional[np.ndarray], h: int):
    """
    COCO keypoints mapping in many pose models:
      1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear (0 is nose)
    We'll use those indices and confidence checks.
    """
    def ok(i: int) -> bool:
        if conf is None:
            return xy[i][0] > 0 and xy[i][1] > 0
        return conf[i] >= KPT_CONF_TH and xy[i][0] > 0 and xy[i][1] > 0

    left_eye_i, right_eye_i = 1, 2
    left_ear_i, right_ear_i = 3, 4

    # Determine side by which eye+ear pair is confident
    has_right = ok(right_eye_i) and ok(right_ear_i)
    has_left = ok(left_eye_i) and ok(left_ear_i)

    if has_right and not has_left:
        orientation = "Right Profile"
        eye = (int(xy[right_eye_i][0]), int(xy[right_eye_i][1]))
        ear = (int(xy[right_ear_i][0]), int(xy[right_ear_i][1] - (h * 0.05)))
        return orientation, ear, eye

    if has_left and not has_right:
        orientation = "Left Profile"
        eye = (int(xy[left_eye_i][0]), int(xy[left_eye_i][1]))
        ear = (int(xy[left_ear_i][0]), int(xy[left_ear_i][1] - (h * 0.05)))
        return orientation, ear, eye

    # If both exist, pick the one with higher confidence sum
    if has_left and has_right and conf is not None:
        right_score = conf[right_eye_i] + conf[right_ear_i]
        left_score = conf[left_eye_i] + conf[left_ear_i]
        if right_score >= left_score:
            orientation = "Right Profile"
            eye = (int(xy[right_eye_i][0]), int(xy[right_eye_i][1]))
            ear = (int(xy[right_ear_i][0]), int(xy[right_ear_i][1] - (h * 0.05)))
        else:
            orientation = "Left Profile"
            eye = (int(xy[left_eye_i][0]), int(xy[left_eye_i][1]))
            ear = (int(xy[left_ear_i][0]), int(xy[left_ear_i][1] - (h * 0.05)))
        return orientation, ear, eye

    return "Unknown", None, None


def get_clinical_note(angle: float, orientation: str, status: str) -> str:
    if client is None:
        return (
            f"Clinical note: {orientation} view shows baseline angle "
            f"{angle}°, status: {status} (tolerance ±{TOLERANCE_DEG}°)."
        )

    prompt = (
        "Write ONE neutral, clinical sentence for a dental positioning "
        "validation report.\n"
        f"Orientation: {orientation}\n"
        f"Baseline angle: {angle} degrees\n"
        f"Tolerance: ±{TOLERANCE_DEG} degrees\n"
        f"Status: {status}\n"
        "Avoid emojis. Avoid casual language."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return (
            f"Clinical note: {orientation} view shows baseline angle "
            f"{angle}°, status: {status} (tolerance ±{TOLERANCE_DEG}°)."
        )



# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "model_loaded": True, "openai_enabled": client is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    _validate_image_upload(file)
    contents = await file.read()
    img = _decode_image(contents)
    h, w, _ = img.shape

    # Run pose model
    try:
        res0 = model(img, verbose=False)[0]
    except Exception as e:
        logger.exception("YOLO inference failed: %s", e)
        raise HTTPException(status_code=500, detail="Pose inference failed.")

    angle = float("nan")
    orientation = "Unknown"
    scale_text = "N/A"
    status = "Unknown"
    note = "Could not detect required landmarks."

    kp = _extract_keypoints(res0)
    if kp is not None:
        xy, conf = kp
        orientation, ear_pt, eye_pt = _pick_profile_points(xy, conf, h)

        if ear_pt and eye_pt:
            angle = _angle_between(ear_pt, eye_pt)
            status = _status_from_angle(angle)

            # Basic scale info (not “real mm”, but useful)
            dist_px = float(np.sqrt((eye_pt[0]-ear_pt[0])**2 + (eye_pt[1]-ear_pt[1])**2))
            # If you have an actual calibration reference, replace this.
            scale_text = f"{dist_px:.1f}px (ear↔eye distance)"

            # Draw annotation
            cv2.line(img, ear_pt, eye_pt, (0, 0, 0), 4)
            cv2.circle(img, ear_pt, 6, (0, 0, 0), -1)
            cv2.circle(img, eye_pt, 6, (0, 0, 0), -1)

            # # Label
            # label = f"{orientation} | {angle}° | {status}"
            # cv2.rectangle(img, (10, 10), (min(10 + 18*len(label), w-10), 55), (255, 255, 255), -1)
            # cv2.putText(img, label, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            note = get_clinical_note(angle, orientation, status)

    return AnalyzeResponse(
        angle=0.0 if np.isnan(angle) else angle,
        orientation=orientation,
        scale=scale_text,
        image=_dataurl_from_bgr(img),
        status=status if status != "Unknown" else "Unknown",
        note=note,
        tolerance_deg=TOLERANCE_DEG,
    )


@app.post("/generate-pdf")
async def generate_pdf(data: GeneratePDFRequest):
    import os, io, base64, tempfile
    from fastapi import HTTPException
    from starlette.responses import StreamingResponse
    from fpdf import FPDF

    def pdf_safe(s: str) -> str:
        """Make text safe for built-in FPDF fonts (latin-1 only)."""
        if s is None:
            return ""
        replacements = {
            "↔": "<->",
            "±": "+/-",
            "°": " deg",
            "“": '"',
            "”": '"',
            "’": "'",
            "–": "-",
            "—": "-",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        # Drop anything else that can't be encoded in latin-1
        return s.encode("latin-1", "ignore").decode("latin-1")

    # Decode base64 image
    if not data.image or "," not in data.image:
        raise HTTPException(status_code=400, detail="Invalid image data URL.")
    b64 = data.image.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.")

    # Save temporary jpg for FPDF
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(img_bytes)
        img_path = tmp.name

    try:
        pdf = FPDF(unit="mm", format="A4")
        pdf.add_page()

        # Header
        pdf.set_fill_color(0, 59, 115)
        pdf.rect(0, 0, 210, 28, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 18)
        pdf.text(12, 18, "PATIENT VISUAL VALIDATION REPORT")

        # Body
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(35)
        pdf.set_font("helvetica", "B", 12)

        pdf.cell(0, 8, pdf_safe(f"Orientation: {data.orientation}"), ln=True)
        pdf.cell(0, 8, pdf_safe(f"Baseline Angle: {data.angle}°"), ln=True)
        pdf.cell(0, 8, pdf_safe(f"Tolerance: ±{TOLERANCE_DEG}°"), ln=True)
        pdf.cell(0, 8, pdf_safe(f"Status: {data.status}"), ln=True)
        pdf.cell(0, 8, pdf_safe(f"Scale: {data.scale}"), ln=True)

        pdf.ln(2)
        pdf.set_font("helvetica", "I", 11)
        pdf.multi_cell(0, 6, pdf_safe(f"Clinical Note: {data.note}"))

        # Image
        pdf.ln(4)
        current_y = pdf.get_y()
        pdf.image(img_path, x=12, y=current_y, w=186)

        # Output bytes (works with fpdf / fpdf2)
        pdf_bytes = pdf.output(dest="S")
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1")

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Patient_Validation_Report.pdf"},
        )

    finally:
        # Cleanup temp file
        try:
            os.remove(img_path)
        except Exception:
            pass
