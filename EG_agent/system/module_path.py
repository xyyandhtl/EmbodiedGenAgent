import os

AGENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGENT_ENV_PATH = os.path.join(AGENT_PATH, "environment")
AGENT_VLMAP_PATH = os.path.join(AGENT_PATH, "vlmap")
AGENT_PROMPT_PATH = os.path.join(AGENT_PATH, "prompts")
AGENT_PLANNING_PATH = os.path.join(AGENT_PATH, "planning")
AGENT_REASONING_PATH = os.path.join(AGENT_PATH, "reasoning")
AGENT_SYSTEM_PATH = os.path.join(AGENT_PATH, "system")
