import os
import traceback
import mimetypes
from typing import Any, Dict, List
from subjective_abstract_data_source_package import SubjectiveOnDemandDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveAnthropicClaudeDataSource(SubjectiveOnDemandDataSource):
    """
    OnDemand data source for Anthropic Claude API interactions.

    Provides access to Claude models including Opus, Sonnet, and Haiku variants
    for text generation and conversation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Read from v2 connection data first, then fall back to v1 params.
        conn = getattr(self, "_connection", {}) or {}

        self.api_key = conn.get("api_key") or self.params.get("api_key", "")
        self.model = conn.get("model") or self.params.get("model", "claude-sonnet-4-5")
        self.temperature = self.params.get("temperature", 0.7)
        self.max_tokens = self.params.get("max_tokens", 4096)
        self.system_prompt = conn.get("system_prompt") or self.params.get("system_prompt", "")
        self.api_base_url = conn.get("api_base_url") or self.params.get("api_base_url", "https://api.anthropic.com")
        self.timeout = self.params.get("timeout", 60)
        self.auto_install_dependencies = bool(
            conn.get("auto_install_dependencies") or self.params.get("auto_install_dependencies", False)
        )
        self._anthropic_available = False
        self._anthropic_import_error = None

        self._normalize_params()
        self._check_dependency()

    @classmethod
    def connection_schema(cls):
        return {
            "api_key": {
                "type": "password",
                "label": "API Key",
                "required": True,
                "placeholder": "sk-ant-...",
            },
            "model": {
                "type": "select",
                "label": "Model",
                "options": [
                    "claude-sonnet-4-5",
                    "claude-opus-4-5",
                    "claude-haiku-4-5",
                ],
                "default": "claude-sonnet-4-5",
            },
            "api_base_url": {
                "type": "url",
                "label": "API Base URL",
                "default": "https://api.anthropic.com",
            },
            "system_prompt": {
                "type": "textarea",
                "label": "System Prompt",
                "description": "Optional system prompt sent with every request",
            },
        }

    def _normalize_params(self) -> None:
        # Ensure sane defaults when upstream config passes zero/invalid values.
        try:
            self.max_tokens = int(self.max_tokens)
        except (TypeError, ValueError):
            self.max_tokens = 4096
        if self.max_tokens <= 0:
            self.max_tokens = 4096

        try:
            self.timeout = int(self.timeout)
        except (TypeError, ValueError):
            self.timeout = 60
        if self.timeout <= 0:
            self.timeout = 60

        if isinstance(self.api_base_url, str) and self.api_base_url:
            base = self.api_base_url.rstrip("/")
            if base.endswith("/v1/messages"):
                base = base[:-12]
            if base.endswith("/v1"):
                base = base[:-3]
            self.api_base_url = base or "https://api.anthropic.com"
        else:
            self.api_base_url = "https://api.anthropic.com"

    def _check_dependency(self) -> None:
        try:
            import anthropic  # noqa: F401
            self._anthropic_available = True
            self._anthropic_import_error = None
        except Exception as e:
            self._anthropic_available = False
            self._anthropic_import_error = e
            BBLogger.log(
                "Anthropic dependency missing. Install with: pip install anthropic "
                "(or install plugin requirements.txt)."
            )

    def _dependency_error_response(self, message: Any) -> Dict[str, Any]:
        hint = "Install with: pip install anthropic"
        if self.auto_install_dependencies:
            hint += " (auto-install not enabled in this environment)"
        return {
            "error": True,
            "error_type": "dependency_error",
            "message": f"The 'anthropic' package is required. {hint}",
            "original_message": message
        }

    def _process_message(self, message: Any) -> Any:
        """
        Process an incoming message using the Anthropic Claude API.

        Args:
            message: The prompt/message to send to Claude

        Returns:
            Dictionary with response data
        """
        if isinstance(message, dict) and message.get("files"):
            return self._process_message_with_files(message)

        # Ensure we have string message
        if isinstance(message, dict):
            message = message.get("content", str(message))
        message = str(message)

        if not self.api_key:
            BBLogger.log("Anthropic API key not configured")
            return {
                "error": True,
                "error_type": "configuration_error",
                "message": "Anthropic API key is required",
                "original_message": message
            }

        if not self._anthropic_available:
            return self._dependency_error_response(message)

        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.api_base_url,
                timeout=self.timeout
            )

            # Build the message request
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "user", "content": message}
                ]
            }

            # Add optional parameters
            if self.temperature is not None:
                request_params["temperature"] = self.temperature

            if self.system_prompt:
                request_params["system"] = self.system_prompt

            BBLogger.log(
                f"Sending request to Claude model: {self.model} "
                f"(max_tokens={self.max_tokens}, temperature={self.temperature}, "
                f"base_url={self.api_base_url}, timeout={self.timeout})"
            )

            response = client.messages.create(**request_params)

            # Extract response content
            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            result = {
                "success": True,
                "response": response_text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "stop_reason": response.stop_reason,
                "original_message": message
            }
            BBLogger.log(
                f"Claude response received (model={response.model}, "
                f"input_tokens={response.usage.input_tokens}, output_tokens={response.usage.output_tokens})"
            )
            return result

        except ImportError:
            BBLogger.log("anthropic package not installed")
            return self._dependency_error_response(message)
        except Exception as e:
            error_type = "api_error"
            error_message = str(e)
            try:
                import anthropic

                if isinstance(e, anthropic.APIConnectionError):
                    error_type = "connection_error"
                    error_message = f"Connection error to {self.api_base_url}: {e}"
                elif isinstance(e, anthropic.APITimeoutError):
                    error_type = "timeout_error"
                    error_message = f"Timeout after {self.timeout}s connecting to {self.api_base_url}"
                elif isinstance(e, anthropic.AuthenticationError):
                    error_type = "auth_error"
                elif isinstance(e, anthropic.RateLimitError):
                    error_type = "rate_limit_error"
                elif isinstance(e, anthropic.BadRequestError):
                    error_type = "bad_request"
                elif isinstance(e, anthropic.APIStatusError):
                    error_type = "api_status_error"
            except Exception:
                pass

            BBLogger.log(f"Error calling Claude API: {e}\n{traceback.format_exc()}")
            return {
                "error": True,
                "error_type": error_type,
                "message": error_message,
                "original_message": message
            }

    def _process_message_with_files(self, message: Dict[str, Any]) -> Any:
        user_text = str(message.get("content") or "")
        files = self._normalize_files(message.get("files"))

        if not self.api_key:
            BBLogger.log("Anthropic API key not configured")
            return {
                "error": True,
                "error_type": "configuration_error",
                "message": "Anthropic API key is required",
                "original_message": user_text
            }

        if not self._anthropic_available:
            return self._dependency_error_response(user_text)

        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.api_base_url,
                timeout=self.timeout
            )

            content_blocks = self._build_claude_content(user_text, files)

            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "user", "content": content_blocks}
                ]
            }

            if self.temperature is not None:
                request_params["temperature"] = self.temperature

            if self.system_prompt:
                request_params["system"] = self.system_prompt

            BBLogger.log(
                f"Sending request to Claude model: {self.model} "
                f"(max_tokens={self.max_tokens}, temperature={self.temperature}, "
                f"base_url={self.api_base_url}, timeout={self.timeout})"
            )

            response = client.messages.create(**request_params)

            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            result = {
                "success": True,
                "response": response_text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "stop_reason": response.stop_reason,
                "original_message": user_text
            }
            BBLogger.log(
                f"Claude response received (model={response.model}, "
                f"input_tokens={response.usage.input_tokens}, output_tokens={response.usage.output_tokens})"
            )
            return result

        except ImportError:
            BBLogger.log("anthropic package not installed")
            return self._dependency_error_response(user_text)
        except Exception as e:
            error_type = "api_error"
            error_message = str(e)
            try:
                import anthropic

                if isinstance(e, anthropic.APIConnectionError):
                    error_type = "connection_error"
                    error_message = f"Connection error to {self.api_base_url}: {e}"
                elif isinstance(e, anthropic.APITimeoutError):
                    error_type = "timeout_error"
                    error_message = f"Timeout after {self.timeout}s connecting to {self.api_base_url}"
                elif isinstance(e, anthropic.AuthenticationError):
                    error_type = "auth_error"
                elif isinstance(e, anthropic.RateLimitError):
                    error_type = "rate_limit_error"
                elif isinstance(e, anthropic.BadRequestError):
                    error_type = "bad_request"
                elif isinstance(e, anthropic.APIStatusError):
                    error_type = "api_status_error"
            except Exception:
                pass

            BBLogger.log(f"Error calling Claude API: {e}\n{traceback.format_exc()}")
            return {
                "error": True,
                "error_type": error_type,
                "message": error_message,
                "original_message": user_text
            }

    def _normalize_files(self, files: Any) -> List[Dict[str, Any]]:
        if not isinstance(files, list):
            return []
        normalized = []
        for item in files:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized

    def _build_claude_content(self, user_text: str, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if user_text:
            content.append({"type": "text", "text": user_text})

        for payload in files:
            mime_type = payload.get("mime_type") or self._guess_mime_type(payload.get("name"))
            data_base64 = payload.get("data_base64")
            if mime_type and mime_type.startswith("image/") and data_base64:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": data_base64
                    }
                })
                continue

            text_block = self._format_file_text(payload)
            if text_block:
                content.append({"type": "text", "text": text_block})

        if not content:
            content.append({"type": "text", "text": ""})

        return content

    def _format_file_text(self, payload: Dict[str, Any]) -> str:
        name = payload.get("name") or "attachment"
        mime_type = payload.get("mime_type") or self._guess_mime_type(name) or "application/octet-stream"
        size = payload.get("size")
        text = payload.get("text")
        if isinstance(text, str) and text:
            return self._truncate_text(
                f"[Attached file: {name} | {mime_type} | {size} bytes]\n{text}"
            )

        data_base64 = payload.get("data_base64")
        if isinstance(data_base64, str) and data_base64:
            snippet = self._truncate_text(data_base64, max_chars=10000)
            return f"[Attached file: {name} | {mime_type} | {size} bytes | base64]\n{snippet}"

        return f"[Attached file: {name} | {mime_type} | {size} bytes | no content provided]"

    def _truncate_text(self, text: str, max_chars: int = 20000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[truncated]"

    def _guess_mime_type(self, name: str) -> str:
        mime_type, _ = mimetypes.guess_type(name)
        return mime_type or "application/octet-stream"

    def get_icon(self) -> str:
        """Return the SVG icon for Anthropic Claude data source."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception as e:
            BBLogger.log(f"Error reading icon file: {e}")

        # Fallback SVG icon (Anthropic-style)
        return '''<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="24" height="24" rx="4" fill="#D97706"/>
            <path d="M12 4L4 20H8L12 12L16 20H20L12 4Z" fill="white"/>
        </svg>'''

    def get_connection_data(self) -> dict:
        """Return connection configuration metadata for Claude API."""
        return {
            "connection_type": "ON_DEMAND",
            "fields": [
                {
                    "name": "connection_name",
                    "type": "text",
                    "label": "Connection Name",
                    "required": True,
                    "description": "A friendly name to identify this Claude connection"
                },
                {
                    "name": "api_key",
                    "type": "password",
                    "label": "Anthropic API Key",
                    "required": True,
                    "description": "Your Anthropic API key from console.anthropic.com"
                },
                {
                    "name": "model",
                    "type": "select",
                    "label": "Model",
                    "required": True,
                    "default": "claude-sonnet-4-5",
                    "description": "Select the Claude model to use",
                    "options": [
                        # Latest Claude 4.5 models
                        {"value": "claude-opus-4-5-20251101", "label": "Claude Opus 4.5 (Most Capable)"},
                        {"value": "claude-sonnet-4-5-20250929", "label": "Claude Sonnet 4.5 (Balanced)"},
                        {"value": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5 (Fastest)"},
                        # Model aliases (auto-updated)
                        {"value": "claude-opus-4-5", "label": "Claude Opus 4.5 (Latest Alias)"},
                        {"value": "claude-sonnet-4-5", "label": "Claude Sonnet 4.5 (Latest Alias)"},
                        {"value": "claude-haiku-4-5", "label": "Claude Haiku 4.5 (Latest Alias)"},
                        # Legacy Claude 4 models
                        {"value": "claude-opus-4-1-20250805", "label": "Claude Opus 4.1 (Legacy)"},
                        {"value": "claude-opus-4-20250514", "label": "Claude Opus 4 (Legacy)"},
                        {"value": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4 (Legacy)"},
                        {"value": "claude-3-7-sonnet-20250219", "label": "Claude Sonnet 3.7 (Legacy)"},
                        {"value": "claude-3-haiku-20240307", "label": "Claude Haiku 3 (Legacy)"}
                    ]
                },
                {
                    "name": "temperature",
                    "type": "number",
                    "label": "Temperature",
                    "required": False,
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Controls randomness in responses (0.0 = deterministic, 1.0 = creative)"
                },
                {
                    "name": "max_tokens",
                    "type": "number",
                    "label": "Max Output Tokens",
                    "required": False,
                    "default": 4096,
                    "min": 1,
                    "max": 64000,
                    "description": "Maximum number of tokens in the response (up to 64K for latest models)"
                },
                {
                    "name": "system_prompt",
                    "type": "textarea",
                    "label": "System Prompt",
                    "required": False,
                    "description": "Optional system instructions to guide Claude's behavior"
                },
                {
                    "name": "api_base_url",
                    "type": "text",
                    "label": "API Base URL",
                    "required": False,
                    "default": "https://api.anthropic.com",
                    "description": "Custom API endpoint (use default unless you have a specific proxy)"
                },
                {
                    "name": "timeout",
                    "type": "number",
                    "label": "Timeout (seconds)",
                    "required": False,
                    "default": 60,
                    "min": 1,
                    "max": 600,
                    "description": "Maximum time to wait for API response"
                }
            ]
        }
