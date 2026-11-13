# Copyright 2026 Keshan Sodimana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-click localhost browser OAuth2 login for google-colab-cli (no terminal input needed)."""

import json
import os
from importlib import resources
from google_auth_oauthlib.flow import InstalledAppFlow
from colab_cli.auth import PUBLIC_SCOPES, _save_token_to_keyring


def main() -> None:
    cfg_text = resources.files("colab_cli").joinpath("oauth_config.json").read_text()
    client_config = json.loads(cfg_text)

    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    print("Opening browser for Google Colab authentication (localhost callback)...", flush=True)
    creds = flow.run_local_server(
        host="localhost",
        port=0,
        open_browser=True,
        prompt="select_account consent",
        access_type="offline",
        authorization_prompt_message="Please complete sign-in in your browser: {url}",
        success_message="Google Colab CLI authenticated! You may close this tab and return to Antigravity.",
    )

    _save_token_to_keyring(creds.to_json())
    print("Saved Colab OAuth credentials to OS keyring!", flush=True)


if __name__ == "__main__":
    main()
