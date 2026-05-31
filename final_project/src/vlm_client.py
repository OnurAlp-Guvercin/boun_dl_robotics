import base64
import json
import re
import time
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from PIL import Image

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_PROMPT_TEMPLATE = """Locate {description} in the tabletop image.
Ignore the robot arm, table border, shadows, and other objects.
Return only JSON: {{"bbox":[x1,y1,x2,y2]}}"""

OBJECT_DESCRIPTIONS = {
    "red_box": "the red cube",
    "red_sphere": "the red sphere",
    "green_box": "the green cube",
    "green_sphere": "the green sphere",
    "blue_box": "the blue cube",
    "blue_sphere": "the blue sphere",
    "yellow_box": "the yellow cube",
    "yellow_sphere": "the yellow sphere",
    "purple_box": "the purple cube",
    "purple_sphere": "the purple sphere",
    "cyan_box": "the cyan cube",
    "cyan_sphere": "the cyan sphere",
}


def _describe_object(object_name: str) -> str:
    """Convert 'red_sphere_3' to the cleanest VLM target phrase."""
    parts = object_name.split("_")
    key = "_".join(parts[:2])
    if key in OBJECT_DESCRIPTIONS:
        return OBJECT_DESCRIPTIONS[key]
    return object_name.replace("_", " ")


def _tensor_to_b64(image: torch.Tensor) -> str:
    """Convert (3, H, W) uint8 tensor → base64-encoded PNG string."""
    arr = image.permute(1, 2, 0).numpy()  # (H, W, 3)
    pil = Image.fromarray(arr.astype("uint8"), "RGB")
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by Qwen3 reasoning mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_bbox(text: str) -> Optional[np.ndarray]:
    """
    Extract [x1, y1, x2, y2] from model output (JSON or inline numbers).
    Returns (cx, cy, w, h) normalised float32, or None on failure.
    """
    text = _strip_thinking(text)
    # Try strict JSON first
    try:
        data = json.loads(text.strip())
        b = data["bbox"]
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except Exception:
        # Fallback: extract 4 consecutive floats/ints from the string
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        if len(nums) < 4:
            return None
        # Take the first 4 numbers found
        try:
            x1, y1, x2, y2 = [float(n) for n in nums[:4]]
        except ValueError:
            return None

    # Normalise coordinate space:
    # Qwen3-VL natively outputs 0-1000, pixel coords would be 0-128
    vals = [x1, y1, x2, y2]
    if any(v > 1.1 for v in vals):
        scale = 1000.0 if max(vals) > 1.5 else 128.0
        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale

    x1, y1, x2, y2 = (
        float(np.clip(x1, 0, 1)),
        float(np.clip(y1, 0, 1)),
        float(np.clip(x2, 0, 1)),
        float(np.clip(y2, 0, 1)),
    )

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = abs(x2 - x1)
    h  = abs(y2 - y1)

    if w < 1e-3 or h < 1e-3:
        return None

    return np.array([cx, cy, w, h], dtype=np.float32)


class VLMClient:
    """Thin wrapper around vLLM's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url:   str   = "http://localhost:8000",
        model_name: str   = "Qwen3",
        max_tokens: int   = 64,
        temperature:float = 0.0,
        timeout:    float = 15.0,
        retries:    int   = 2,
    ) -> None:
        if not _HAS_REQUESTS:
            raise ImportError("Install 'requests': pip install requests")
        self.url        = base_url.rstrip("/") + "/v1/chat/completions"
        self.model      = model_name
        self.max_tokens = max_tokens
        self.temperature= temperature
        self.timeout    = timeout
        self.retries    = retries

    def get_bbox(
        self,
        image: torch.Tensor,
        object_name: str,
    ) -> Optional[np.ndarray]:
        """
        Parameters
        ----------
        image       : (3, H, W) uint8 torch tensor
        object_name : human-readable name, e.g. "red_box_0"

        Returns
        -------
        np.ndarray of shape (4,) = (cx, cy, w, h) normalised [0,1], or None.
        """
        b64   = _tensor_to_b64(image)
        prompt = _PROMPT_TEMPLATE.format(
            name=object_name,
            description=_describe_object(object_name),
        )

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                bbox = _parse_bbox(text)
                if bbox is not None:
                    return bbox
            except Exception:
                if attempt < self.retries:
                    time.sleep(0.5)
        return None

    def is_available(self) -> bool:
        """Ping the server and return True if it responds."""
        try:
            r = requests.get(
                self.url.replace("/v1/chat/completions", "/health"),
                timeout=3.0,
            )
            return r.status_code < 500
        except Exception:
            return False
