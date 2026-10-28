"""Command line: chatgpt_as_api login | refresh | whoami | models | test"""
import argparse
import json
import sys

from .auth import AuthStore, AuthError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chatgpt_as_api", description=__doc__)
    ap.add_argument("--auth", help="token file (default ~/.config/chatgpt_as_api/auth.json or $CHATGPT_AS_API_AUTH)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("login", help="sign in with ChatGPT (opens nothing: prints a URL to open)")
    p.add_argument("--port", type=int, default=1455)
    p.add_argument("--name", default="chatgpt_as_api", help="app name shown on the consent screen")
    sub.add_parser("refresh", help="refresh the access token now")
    sub.add_parser("whoami", help="signed-in account and granted scopes")
    sub.add_parser("models", help="models available to the plan token")
    p = sub.add_parser("test", help="one Responses request")
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--web-search", action="store_true")
    p.add_argument("--prompt", default="In one sentence: what is the Responses API?")
    args = ap.parse_args(argv)

    auth = AuthStore(args.auth)
    try:
        if args.cmd == "login":
            auth.login(agent_name=args.name, port=args.port)
        elif args.cmd == "refresh":
            auth.refresh()
            print("refreshed; expires in", auth.data["tokens"].get("expires_in"), "s")
        elif args.cmd == "whoami":
            claims = auth.id_claims()
            print("email:", claims.get("email"), "| client_id:", auth.data.get("client_id"))
            print("scopes:", " ".join(auth.scopes), "| plan usage:", auth.plan_usage)
        elif args.cmd == "models":
            from .client import OpenAI
            for m in OpenAI(auth_path=args.auth).models.list():
                print(f"{m['id']:22s} {m.get('display_name', ''):18s} ctx {m.get('context_window')}  {m.get('description', '')}")
        elif args.cmd == "test":
            from .client import OpenAI
            client = OpenAI(auth_path=args.auth)
            kwargs = {"model": args.model, "input": args.prompt}
            if args.web_search:
                kwargs.update(tools=[{"type": "web_search"}], tool_choice="required")
            r = client.responses.create(**kwargs)
            print(r.output_text)
            print("usage:", json.dumps(r.usage.model_dump(exclude_none=True) if r.usage else None))
    except AuthError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
