import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Union

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
    High‑level wrapper for prompt+image → LLM.
    Removed template_infer support. Added simple memory (chat history).
    """

    def __init__(self) -> None:
        self.client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
        self.system_prompt = ""
        # simple memory as list of messages: {"role": "user"/"assistant", "content": str or list}
        self.memory: list[dict] = []

        logger.info("VLMInference initialized (no template_infer).")

    # ---------------------------------------------------------------------
    # Memory helpers
    # ---------------------------------------------------------------------
    def add_memory(self, role: str, content) -> None:
        """Append a message to the internal memory. content can be a str or structured payload."""
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        self.memory.append({"role": role, "content": content})

    def clear_memory(self) -> None:
        """Clear stored memory/history."""
        self.memory.clear()

    def get_memory(self) -> list:
        """Return a shallow copy of memory."""
        return list(self.memory)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def add_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt += system_prompt
        logger.info(f"Updated system prompt: \n{self.system_prompt}")
    
    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        logger.info(f"Updated system prompt: \n{self.system_prompt}")

    def add_system_prompt_file(self, system_prompt_file: str) -> None:
        self.system_prompt += self._load_prompt(system_prompt_file)
        logger.info(f"Updated system prompt: \n{self.system_prompt}")

    def infer(self, 
              text: str, 
              image: Union[str, Path, Image.Image, np.ndarray, None] = None, 
              record_memory: bool = False) -> str:
        """
        Perform inference. If record_memory is True, the user prompt and the assistant reply
        will be appended to internal memory after a successful call.
        """
        try:
            image_url = self._prepare_image(image) if image else ""
            # build prompt text (no template handling)
            prompt = text

            reply = self._call_llm(image_url, prompt, include_memory=True)

            # optionally record into memory
            if record_memory:
                # store user turn (text + optional image_url) and assistant reply
                # if store image in memory:
                # user_content = [{"type": "text", "text": prompt}] if image_url else prompt
                # if image_url:
                #     user_content.append({"type": "image_url", "image_url": {"url": image_url}})
                self.add_memory("user", prompt)
                self.add_memory("assistant", reply)

            return reply
        except Exception as exc:
            logger.exception("Inference failed: %s", exc)
            return str(exc)

    # ---------------------------------------------------------------------
    # Prompt management
    # ---------------------------------------------------------------------

    def _load_prompt(self, prompt_file: str) -> str:
        return Path(PROMPT_DIR / prompt_file).read_text(encoding="utf-8")

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

    def _call_llm(self, image_url: str, prompt: str, include_memory: bool = True) -> str:
        t0 = time.time()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # include past memory if requested
        if include_memory and self.memory:
            # memory items are already in the form {"role":..., "content":...}
            # for msg in self.memory:
            #     messages.append({"role": msg["role"], "content": msg["content"]})
            messages += self.memory       

        # current user message (with optional image)
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
        # assistant content extraction
        return resp.choices[0].message.content

###############################################################################
# Main execution                                                              #
###############################################################################

if __name__ == "__main__":
    vlm = VLMInference()
    vlm.add_system_prompt_file("_test/system.txt")
    # Text-only inference
    result2 = vlm.infer("我要找显示器", None, record_memory=True)
    print(result2)

    # Image-based inference (and record in memory)
    result = vlm.infer("我要找显示器", "/home/lenovo/Opensources/vlm_robot_agent/img/1_center.jpg", 
                       record_memory=True)
    print(result)
