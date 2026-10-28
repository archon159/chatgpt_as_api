# chatgpt_as_api

Use the OpenAI Responses API on your ChatGPT plan (Plus/Pro) instead of API credits, through
OpenAI's official **Sign in with ChatGPT** flow for open-source apps. The package is a drop-in for
the `openai` SDK: swap the import and keep your code.

```python
from chatgpt_as_api import OpenAI   # instead of: from openai import OpenAI

client = OpenAI()
r = client.responses.create(model="gpt-5.6-luna", input="Summarize the Responses API in one sentence.")
print(r.output_text)

# built-in web search works, including forcing it
r = client.responses.create(model="gpt-5.6-luna", input=messages,
                            tools=[{"type": "web_search"}], tool_choice="required")
```

Only official endpoints are used (`https://api.openai.com/v1/responses`, `/v1/models`), with the
OAuth token OpenAI issues for this purpose. Usage counts against the signed-in account's plan limits.

## Install and sign in

```bash
pip install -e .
chatgpt_as_api login      # prints a URL; open it in a browser, sign in, approve
chatgpt_as_api models     # models available to your plan
chatgpt_as_api test --model gpt-5.6-luna --web-search
```

The login uses a loopback callback on `127.0.0.1:1455`. On a remote machine, forward the port first
(`ssh -L 1455:127.0.0.1:1455 host`) and open the URL in your local browser. Tokens are stored in
`~/.config/chatgpt_as_api/auth.json` (or `$CHATGPT_AS_API_AUTH`) and refreshed automatically; the
refresh token lasts 30 days, after which `login` is needed again.

## What the plan route allows

The route has rules (OpenAI's [preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations));
the client applies them for you:

| rule | handled as |
|---|---|
| `store` must be false, `stream` must be true | set automatically; a non-streaming `create` consumes the stream and returns the final `Response` |
| no `system` / `developer` messages | moved into `instructions` |
| `input` must be a list of items | a string becomes one user message |
| `max_output_tokens`, `metadata`, `background`, `max_tool_calls`, `previous_response_id`, `conversation` rejected | dropped with a warning |
| models limited to the account's catalog | `client.models.list()` shows them; API-only models such as `gpt-4o-mini` are not available |
| `chat.completions`, file search, code interpreter, image generation, hosted MCP not available | not provided |

Usage-limit errors arrive as `429 subscription_sharing_usage_limit_exceeded`; eligibility errors as
`403 subscription_sharing_user_not_eligible`.

## Async

```python
from chatgpt_as_api import AsyncOpenAI
client = AsyncOpenAI()
r = await client.responses.create(model="gpt-5.6-luna", input="hello")
```

## License

MIT.
