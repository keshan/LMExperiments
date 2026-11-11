"""One-click localhost browser OAuth2 login for google-colab-cli (no terminal input needed)."""

import json
import os
from importlib import resources
from google_auth_oauthlib.flow import InstalledAppFlow
from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH


def main() -> None:
    cfg_text = resources.files("colab_cli").joinpath("oauth_config.json").read_text()
    client_config = json.loads(cfg_text)

    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    print("Opening browser for Google Colab authentication (localhost callback)...", flush=True)
    creds = flow.run_local_server(
        host="localhost",
        port=0,
        open_browser=True,
        prompt="consent",
        access_type="offline",
        authorization_prompt_message="Please complete sign-in in your browser: {url}",
        success_message="Google Colab CLI authenticated! You may close this tab and return to Antigravity.",
    )

    os.makedirs(os.path.dirname(TOKEN_CONFIG_PATH), exist_ok=True)
    with open(TOKEN_CONFIG_PATH, "w") as f:
        f.write(creds.to_json())
    print(f"Saved Colab OAuth credentials to {TOKEN_CONFIG_PATH}", flush=True)


if __name__ == "__main__":
    main()
