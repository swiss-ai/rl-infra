"""Turn a Messages request the probe recorded into the chat-completions body Gym's vllm_model sends NeMo-RL, for
check B (backends/nemo-rl/apertus2/check_apertus2_tools.sh, EXTRA_REQUESTS).

    python to_chat.py <gym checkout> <first-request.json> <out.json> [max_tokens]

The path is the model server's: AnthropicConverter (Messages to Responses), then VLLMConverter with the training
config's flags (vllm_model_for_training.yaml: return_token_id_information, uses_reasoning_parser), then vllm_model's
late_system_messages_as_user (Gym 0052, set by claude-code-sc.yaml: Apertus 2 takes system messages only first). A
first request has no assistant turn, so vllm_model's rewriting of assistant messages does not apply. `max_tokens` is the per-turn
limit as the run's sampling_overrides set it (policy.generation.max_new_tokens), since check B checks that the prompt
plus it fit the context. The output holds Claude Code's system prompt: keep it out of git.
"""

import json
import sys
from pathlib import Path


gym, src, dst = sys.argv[1:4]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else None
sys.path.insert(0, gym)

from nemo_gym.anthropic_converter import AnthropicConverter  # noqa: E402
from nemo_gym.responses_converter import VLLMConverter  # noqa: E402
from responses_api_models.vllm_model.app import late_system_messages_as_user  # noqa: E402


body = json.loads(Path(src).read_text())
params = AnthropicConverter().anthropic_request_to_responses(body)
chat = VLLMConverter(return_token_id_information=True, uses_reasoning_parser=True).responses_to_chat_completion_create_params(
    params
)
chat = chat.model_dump(exclude_none=True) if hasattr(chat, "model_dump") else dict(chat)
late_system_messages_as_user(chat["messages"])
if max_tokens is not None:
    chat["max_tokens"] = max_tokens
Path(dst).write_text(json.dumps(chat, indent=1))
roles = [m.get("role") for m in chat.get("messages", [])]
names = [t.get("function", {}).get("name") for t in chat.get("tools") or []]
print(f"{dst}: roles {roles}, tools {names}, max_tokens {chat.get('max_tokens')}")
