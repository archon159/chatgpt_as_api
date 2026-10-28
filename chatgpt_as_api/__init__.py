"""Use the OpenAI Responses API on your ChatGPT plan, through the official Sign in with ChatGPT flow.

    from chatgpt_as_api import OpenAI
    client = OpenAI()
    r = client.responses.create(model="gpt-5.6-luna", input="hello")
"""
from .auth import AuthError, AuthStore
from .client import AsyncOpenAI, OpenAI, PlanRouteError

__all__ = ["OpenAI", "AsyncOpenAI", "AuthStore", "AuthError", "PlanRouteError"]
__version__ = "0.1.0"
