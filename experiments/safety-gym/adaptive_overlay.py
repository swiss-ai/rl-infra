"""Pinned, feature-only source overlays for the native inference/runtime boundary."""
import hashlib

EXPECTED_GYM_VLLM_APP_SHA256 = "f2a843069849e250d35444bad75f29dccd52b1cc0d668b4b31f32623f12b205d"

_NATIVE_MEASURE_METHOD = '''    async def measure_context_native(self, request: Request, body: dict = Body()) -> dict[str, int]:
        """Measure complete explicit history through this model's actual worker tokenizer.

        This route deliberately has no capture lineage or parent-resolution state. The caller
        sends every original Responses input item, including native policy outputs, explicitly.
        """
        if self.config.forward_unvalidated_reasoning_effort:
            body = _set_aside_unlisted_reasoning_effort(request, body)
        body = _validate_responses_params(body)
        if body.previous_response_id or body.conversation:
            raise HTTPException(status_code=422, detail="measurement requires explicit non-streaming history")
        client = self._resolve_client(request)
        try:
            chat = self._responses_to_chat_request(body, request)
            processed = await self._prepare_chat_request(request, chat)
        except (ValueError, NotImplementedError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        result = await client.create_tokenize(**self._get_tokenize_chat_body(processed))
        count = result.get("count")
        if type(count) is not int and isinstance(result.get("tokens"), list):
            count = len(result["tokens"])
        if type(count) is not int or count < 0:
            raise HTTPException(status_code=502, detail="worker did not acknowledge exact context measurement")
        try:
            from nemo_rl.data.apertus2_format import _get_encoding
            encoding = _get_encoding(self.config.tokenizer or self.config.model)
            wait_token_id = encoding.controls.wait
        except Exception as error:
            raise HTTPException(status_code=502, detail="native Apertus wait-token identity is unavailable") from error
        if type(wait_token_id) is not int or wait_token_id < 0:
            raise HTTPException(status_code=502, detail="native Apertus wait-token identity is invalid")
        return {"prompt_token_count": count, "wait_token_id": wait_token_id, "ng_context_measured": True}
'''


def render_native_context_measure(source: str) -> str:
    """Add only the full-history measurement route to the reviewed Gym adapter SHA."""
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_GYM_VLLM_APP_SHA256:
        raise ValueError(f"Unknown Gym vLLM adapter source fingerprint: {digest}")

    key_field_anchor = "    api_key: str\n"
    if source.count(key_field_anchor) != 1:
        raise ValueError("Unknown VLLM model credential field")
    source = source.replace(
        key_field_anchor,
        '    api_key: str = Field(default="", repr=False, exclude=True)\n'
        '    api_key_file: Optional[str] = None\n'
        '    api_key_env_name: Optional[str] = None\n',
        1,
    )
    key_loader = '''
from pathlib import Path as _AdaptivePath

def _adaptive_private_key(path: str, variable_name: str) -> str:
    private_path = _AdaptivePath(path)
    if not private_path.is_file() or (private_path.stat().st_mode & 0o077):
        raise RuntimeError("private API-key file must exist and have mode 0600")
    matches = []
    for line in private_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == variable_name:
            matches.append(value.strip().strip(chr(34) + chr(39)))
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError("private API-key variable is missing or duplicated")
    return matches[0]
'''
    insertion = 'class VLLMModelConfig(BaseResponsesAPIModelConfig):\n'
    if source.count(insertion) != 1:
        raise ValueError("Unknown VLLM config class boundary")
    source = source.replace(insertion, key_loader + "\n" + insertion, 1)
    post_init = '''    def model_post_init(self, context):
        self._post_init()
        return super().model_post_init(context)
'''
    post_init_replacement = '''    def model_post_init(self, context):
        if self.config.api_key_file:
            if not self.config.api_key_env_name:
                raise RuntimeError("private API-key variable name is required")
            object.__setattr__(
                self,
                "_adaptive_runtime_api_key",
                _adaptive_private_key(self.config.api_key_file, self.config.api_key_env_name),
            )
        self._post_init()
        return super().model_post_init(context)
'''
    if source.count(post_init) != 1:
        raise ValueError("Unknown VLLM model initialization boundary")
    source = source.replace(post_init, post_init_replacement, 1)
    source = source.replace(
        "api_key=self.config.api_key,",
        'api_key=getattr(self, "_adaptive_runtime_api_key", self.config.api_key),',
    )

    route_anchor = (
        '        if config.token_id_capture.parent_resolution == "source_history":\n'
        '            app.post("/context/measure")(self.measure_context)\n'
    )
    if source.count(route_anchor) != 1:
        raise ValueError("Unknown native context-measure route registration")
    source = source.replace(
        route_anchor,
        route_anchor + '        app.post("/context/measure_native")(self.measure_context_native)\n',
        1,
    )

    routes_anchor = 'frozenset({("POST", "/context/measure")})'
    if source.count(routes_anchor) != 1:
        raise ValueError("Unknown non-generating route declaration")
    source = source.replace(
        routes_anchor,
        'frozenset({("POST", "/context/measure"), ("POST", "/context/measure_native")})',
        1,
    )

    method_anchor = '    def get_converter(self) -> "VLLMConverter":\n'
    if source.count(method_anchor) != 1:
        raise ValueError("Unknown native measurement method boundary")
    source = source.replace(method_anchor, _NATIVE_MEASURE_METHOD + "\n" + method_anchor, 1)
    compile(source, "vllm_model_app.py", "exec")
    return source


EXPECTED_GYM_OPENAI_CLIENT_SHA256 = "7f9190b72ce2aeac41e6a3d750a990677fbdcfe63bf4cc0b1bc687fc2d944748"

def render_native_client_redaction(source: str) -> str:
    """Exclude the runtime credential from the Gym SDK client's repr and model dumps."""
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_GYM_OPENAI_CLIENT_SHA256:
        raise ValueError(f"Unknown Gym OpenAI client source fingerprint: {digest}")
    anchor = "    api_key: str\n"
    if source.count(anchor) != 1:
        raise ValueError("Unknown Gym SDK credential field")
    source = source.replace(
        anchor,
        "    api_key: str = Field(..., repr=False, exclude=True)\n",
        1,
    )
    compile(source, "openai_client.py", "exec")
    return source
