"""Configure Hugging Face auth from /root/.hf_token on the remote instance."""
from pathlib import Path

from huggingface_hub import login

token = Path("/root/.hf_token").read_text(encoding="utf-8").strip()
login(token=token)
print("HF login ok")

bashrc = Path("/root/.bashrc")
lines = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
if "HF_TOKEN" not in lines:
    bashrc.write_text(
        lines
        + "\nexport HF_TOKEN=$(cat /root/.hf_token 2>/dev/null)\n"
        + "export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\n",
        encoding="utf-8",
    )

followup = Path("/root/tibetan-metadata-detector/run_after_roberta.sh")
if followup.exists():
    text = followup.read_text(encoding="utf-8")
    if "HF_TOKEN" not in text:
        followup.write_text(
            text.replace(
                "#!/bin/bash\n",
                "#!/bin/bash\n"
                "export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null)\n"
                "export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN\n",
            ),
            encoding="utf-8",
        )
