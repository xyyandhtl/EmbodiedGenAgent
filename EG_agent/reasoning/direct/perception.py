from typing import Dict, Any, Union
from pathlib import Path

from PIL import Image
import numpy as np

from reasoning.simple.inference import VLMInference  # type: ignore

__all__ = ["Perception", "Observation"]

# ---------------------------------------------------------------------------
# Typed alias for an observation returned by Perception.perceive()
Observation = Dict[str, Any]


class Perception:
    """High‑level perception interface.

    Parameters
    ----------
    goal_text : str
        The current high‑level goal – is baked into the prompt of each
        *VLMInference* instance.
    provider : str, default "openai"
        Inference backend; passed straight to :class:`VLMInference`.
    history_size : int, default 10
        Length of the circular buffer inside each :class:`VLMInference`.
    """

    def __init__(self, *, goal_text: str, provider: str = "openai", history_size: int = 10):
        self._navigation_engine = VLMInference(
            goal=goal_text,
            provider=provider,
            history_size=history_size,
        )
        self._interaction_engine = VLMInference(
            goal=goal_text,
            provider=provider,
            history_size=history_size,
        )

    # ------------------------------------------------------------------
    def perceive(
        self,
        img: Union[str, Path, Image.Image, np.ndarray],
        *,
        mode: str = "navigation",  # "navigation" | "interaction"
    ) -> Observation:
        """Run VLM inference and standardize its output.

        *No* domain logic lives here – we simply normalize the JSON so downstream
        modules need not worry about slight variations in the model output.
        """
        engine = self._navigation_engine if mode == "navigation" else self._interaction_engine
        result = engine.infer(img)

        # Flatten / coerce the TypedDict coming from VLMInference into a simple dict.
        observation: Observation = {
            "status": getattr(result["status"], "value", result["status"]),
            "description": result.get("description", ""),
            "obstacles": result.get("obstacles", []),
            "current_environment_type": result.get("current_environment_type", "UNKNOWN_ENV"),
            "suggested_actions": result.get("actions", []),
            # quick boolean for convenience ↓
            "goal_observed": any(
                str(a.get("Goal_observed", "False")).lower() == "true" for a in result.get("actions", [])
            ),
        }
        return observation

    def is_visible(self, img: Union[str, Path, Image.Image, np.ndarray]) -> bool:
        """Returns True if the target is observed in the image."""
        observation = self.perceive(img, mode="navigation")
        return observation["goal_observed"]
    
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from pprint import pprint
    from PIL import Image

    parser = argparse.ArgumentParser(description="Quick test for Perception")
    parser.add_argument("--image", type=Path, required=True, help="Path to the image")
    parser.add_argument("--goal", default="Find the person", help="Current goal")  # default is person

    parser.add_argument("--mode", choices=["navigation", "interaction"],
                        default="navigation", help="Inference mode")
    args = parser.parse_args()

    # Instantiate
    perceptor = Perception(goal_text=args.goal)

    # Load image
    img = Image.open(args.image)

    # Infer
    obs = perceptor.perceive(img, mode=args.mode)

    print("\n─ Observation ─")
    pprint(obs, sort_dicts=False)

    # Check visibility
    is_person_visible = perceptor.is_visible(img)

    print(f"\n🧑‍🦱 Is the person (goal '{args.goal}') visible?: {'✅ YES' if is_person_visible else '❌ NO'}")
