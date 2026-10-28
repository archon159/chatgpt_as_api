"""OpenAI SDK clients that run Responses API requests on a ChatGPT plan.

`OpenAI` / `AsyncOpenAI` subclass the official clients: the bearer token is the Sign in with
ChatGPT access token (refreshed automatically) and `responses.create` is adjusted to the rules of
the plan route (developers.openai.com/siwc/token-sharing-open-source/preview-limitations):

- `store` must be false and `stream` true: a non-streaming `create` call is served by consuming the
  stream and returning the final `Response`, so callers see the usual object.
- No system-role messages: they are moved into `instructions`.
- `input` must be a list of items: a plain string becomes one user message.
- Some fields are rejected and are dropped with a warning (`max_output_tokens`, `metadata`,
  `background`, `max_tool_calls`, `previous_response_id`, `conversation`).

Only `/v1/responses` and `/v1/models` are available on this route; `chat.completions` is not.
"""
from __future__ import annotations

import warnings
from typing import Any

import httpx
import openai

from .auth import AuthStore

DROPPED_FIELDS = ("max_output_tokens", "metadata", "background", "max_tool_calls", "previous_response_id", "conversation")
SYSTEM_ROLES = ("system", "developer")


class PlanRouteError(openai.OpenAIError):
    pass


def _as_list(value: Any) -> list:
    return [{"role": "user", "content": value}] if isinstance(value, str) else list(value)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


def prepare_request(kwargs: dict) -> dict:
    """Rewrite `responses.create` arguments to what the plan route accepts."""
    kwargs = dict(kwargs)
    for field in DROPPED_FIELDS:
        if field in kwargs and kwargs[field] is not None and kwargs[field] is not openai.omit:
            warnings.warn(f"chatgpt_as_api: `{field}` is not supported on the ChatGPT plan route and was dropped", stacklevel=3)
        kwargs.pop(field, None)
    items = _as_list(kwargs.get("input", []))
    system_parts = [_message_text(m.get("content")) for m in items if isinstance(m, dict) and m.get("role") in SYSTEM_ROLES]
    if system_parts:
        kwargs["input"] = [m for m in items if not (isinstance(m, dict) and m.get("role") in SYSTEM_ROLES)]
        given = kwargs.get("instructions")
        kwargs["instructions"] = "\n\n".join(p for p in ([given] if isinstance(given, str) and given else []) + system_parts)
    else:
        kwargs["input"] = items
    kwargs["store"] = False
    return kwargs


class _Collector:
    """Builds the final `Response` from a stream: on the plan route the `response.completed` event
    carries an empty `output` once tools were used, so the output items are taken from the
    `response.output_item.done` events instead."""

    def __init__(self):
        self.items = []
        self.final = None

    def feed(self, event) -> None:
        kind = event.type
        if kind == "response.output_item.done":
            self.items.append(event.item)
        elif kind in ("response.completed", "response.incomplete"):
            self.final = event.response
        elif kind == "response.failed":
            err = event.response.error
            raise PlanRouteError(f"response failed: {getattr(err, 'code', None)}: {getattr(err, 'message', err)}")

    def result(self):
        if self.final is None:
            raise PlanRouteError("stream ended without a response.completed event")
        if not self.final.output and self.items:
            return self.final.model_copy(update={"output": list(self.items)})
        return self.final


class _PlanResponses:
    """`client.responses` replacement: same `create` and `stream` names as the SDK."""

    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        kwargs = prepare_request(kwargs)
        if kwargs.get("stream"):
            return self._inner.create(**kwargs)
        kwargs["stream"] = True
        collector = _Collector()
        for event in self._inner.create(**kwargs):
            collector.feed(event)
        return collector.result()

    def stream(self, **kwargs):
        kwargs = prepare_request(kwargs)
        kwargs.pop("stream", None)
        return self._inner.stream(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _AsyncPlanResponses(_PlanResponses):
    async def create(self, **kwargs):
        kwargs = prepare_request(kwargs)
        if kwargs.get("stream"):
            return await self._inner.create(**kwargs)
        kwargs["stream"] = True
        collector = _Collector()
        async for event in await self._inner.create(**kwargs):
            collector.feed(event)
        return collector.result()


class _PlanModels:
    """`client.models.list()`: the plan route answers `{"models": [{"slug": ...}]}`, not the SDK's page shape."""

    def __init__(self, auth: AuthStore, base_url: str):
        self._auth, self._base_url = auth, base_url.rstrip("/")

    def list(self) -> list[dict]:
        r = httpx.get(f"{self._base_url}/models", headers={"Authorization": f"Bearer {self._auth.access_token()}"}, timeout=60)
        if r.status_code != 200:
            raise PlanRouteError(f"GET /models failed ({r.status_code}): {r.text[:500]}")
        body = r.json()
        models = body.get("models") or body.get("data") or []
        return [{"id": m.get("slug") or m.get("id"), **m} for m in models]

    def ids(self) -> list[str]:
        return [m["id"] for m in self.list()]


class _AsyncPlanModels(_PlanModels):
    async def list(self) -> list[dict]:  # type: ignore[override]
        async with httpx.AsyncClient(timeout=60) as http:
            r = await http.get(f"{self._base_url}/models", headers={"Authorization": f"Bearer {self._auth.access_token()}"})
        if r.status_code != 200:
            raise PlanRouteError(f"GET /models failed ({r.status_code}): {r.text[:500]}")
        body = r.json()
        models = body.get("models") or body.get("data") or []
        return [{"id": m.get("slug") or m.get("id"), **m} for m in models]

    async def ids(self) -> list[str]:  # type: ignore[override]
        return [m["id"] for m in await self.list()]


def _check(auth: AuthStore) -> None:
    if not auth.logged_in:
        raise PlanRouteError(f"not logged in: run `chatgpt_as_api login` (token file {auth.path})")
    if not auth.plan_usage:
        raise PlanRouteError("the saved token has no ChatGPT plan usage scope; run `chatgpt_as_api login` again")


class OpenAI(openai.OpenAI):
    """Drop-in for `openai.OpenAI`; requests run on the signed-in user's ChatGPT plan."""

    def __init__(self, *, auth_path: str | None = None, **kwargs):
        self.auth = AuthStore(auth_path)
        _check(self.auth)
        kwargs.setdefault("default_headers", {}).setdefault("User-Agent", "chatgpt_as_api")
        super().__init__(api_key=self.auth.access_token, **kwargs)  # callable: refreshed on every request
        self.responses = _PlanResponses(super().responses)
        self.models = _PlanModels(self.auth, str(self.base_url))


class AsyncOpenAI(openai.AsyncOpenAI):
    """Drop-in for `openai.AsyncOpenAI`; requests run on the signed-in user's ChatGPT plan."""

    def __init__(self, *, auth_path: str | None = None, **kwargs):
        self.auth = AuthStore(auth_path)
        _check(self.auth)
        kwargs.setdefault("default_headers", {}).setdefault("User-Agent", "chatgpt_as_api")

        async def provider() -> str:  # the async client awaits its api_key provider
            return self.auth.access_token()

        super().__init__(api_key=provider, **kwargs)
        self.responses = _AsyncPlanResponses(super().responses)
        self.models = _AsyncPlanModels(self.auth, str(self.base_url))
