import base64
import io
import json
import logging
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union
from importlib import resources

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw

###############################################################################
# Logging & utility helpers                                                   #
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vlm_inference")

VLM_MODEL_NAME = "InternVL3"  # Replace with your local model name
PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"

def annotate_tercios(img: Image.Image, color: tuple[int, int, int, int] = (255, 0, 255, 80)) -> Image.Image:
    """
    Overlay two vertical lines that split the image into thirds (Left/Center/Right).
    Returns a *new* RGBA image (does not modify the original).
    """
    w, h = img.size
    x1, x2 = w / 3, 2 * w / 3

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.line([(x1, 0), (x1, h)], fill=color, width=5)
    draw.line([(x2, 0), (x2, h)], fill=color, width=5)

    return Image.alpha_composite(img.convert("RGBA"), overlay)

def pil_to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    """
    Convert a PIL Image to a data‑URL (base64). Auto‑converts RGBA→RGB for JPEG.
    """
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    buff = io.BytesIO()
    img.save(buff, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buff.getvalue()).decode()}"

###############################################################################
# Main class                                                                  #
###############################################################################

class VLMInference:
    """
    High‑level wrapper that embeds: prompt → image → call LLM → parse JSON.
    """

    def __init__(
        self,
        template_infer: bool = False,
    ) -> None:
        self.client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
        self.system_prompt = ""
        self.user_prompt = ""
        self.template_infer = template_infer

        logger.info(f"template_infer: {self.template_infer}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def add_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt += system_prompt
        logger.info(f"Updated system prompt: \n{self.system_prompt}")

    def add_system_prompt_file(self, system_prompt_file: str) -> None:
        self.system_prompt += self._load_prompt(system_prompt_file)
        logger.info(f"Updated system prompt: \n{self.system_prompt}")

    def add_user_prompt_template(self, user_prompt_file: str) -> None:
        self.user_prompt_template = self._load_prompt(user_prompt_file)
        logger.info(f"Updated user prompt: {self.user_prompt_template}")

    def infer(self, text: str, image: Union[str, Path, Image.Image, np.ndarray, None] = None) -> str:
        try:
            image_url = self._prepare_image(image) if image else None
            prompt = self._format_prompt(text) if self.template_infer else text

            # Log consolidated information
            logger.info(f"Infer called | text: {prompt} | image: {image_url is not None} | template_infer: {self.template_infer}")

            return self._call_llm(image_url, prompt)
        except Exception as exc:
            logger.exception("Inference failed: %s", exc)
            return str(exc)

    # ---------------------------------------------------------------------
    # Prompt management
    # ---------------------------------------------------------------------

    def _load_prompt(self, prompt_file: str) -> str:
        return Path(PROMPT_DIR / prompt_file).read_text(encoding="utf-8")

    def _format_prompt(self, text: str) -> str:
        return self.user_prompt_template.format(text=text)

    # ---------------------------------------------------------------------
    # Image helpers
    # ---------------------------------------------------------------------

    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return pil_to_data_url(img)

    # ---------------------------------------------------------------------
    # OpenAI call
    # ---------------------------------------------------------------------

    def _call_llm(self, image_url: str, prompt: str) -> str:
        t0 = time.time()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if image_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            })
        else:
            messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=VLM_MODEL_NAME,
            max_tokens=2048,
            messages=messages,
        )
        logger.info("LLM latency %.2fs", time.time() - t0)
        return resp.choices[0].message.content

###############################################################################
# Main execution                                                              #
###############################################################################

if __name__ == "__main__":
    vlm = VLMInference(template_infer=True)
    vlm.add_system_prompt_file("_test/system.txt")
    vlm.add_user_prompt_template("_test/user.txt")
    # Text-only inference
    result2 = vlm.infer("显示器", None)
    print(result2)

    # Image-based inference
    result = vlm.infer("显示器", "/home/lenovo/Opensources/vlm_robot_agent/img/1_center.jpg")
    print(result)
