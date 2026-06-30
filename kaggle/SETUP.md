# Kaggle Auto-Runner / Pusher — Setup

This is the configuration for the autonomous GPU loop. Once set up, the flow is:

```
Claude (local)  --push code+configs-->  GitHub (branch: research)
                                              |
                                              v
Kaggle notebook (GPU) --poll--> pull --> run pending configs --> push results to outputs/
                                              |
                                              v
Claude (next session) <--read outputs/-- GitHub   (interprets, writes research-log, queues next config)
```

The runner only ever writes under `outputs/`. Claude only writes code, `configs/`, and
`notes/`. Disjoint paths ⇒ no merge conflicts.

---

## One-time setup

### 1. Create a GitHub Personal Access Token (PAT)
- GitHub → Settings → Developer settings → **Personal access tokens** → *Fine-grained*.
- Repository access: only `ml-researcher123/thesis-code`.
- Permissions: **Contents: Read and write**.
- Copy the token (starts with `github_pat_...`).

### 2. Store it as a Kaggle Secret
- New/!existing Kaggle Notebook → **Add-ons → Secrets → Add secret**.
- Label: `GITHUB_TOKEN`, Value: the PAT.
- (Secrets are per-notebook; attach in the notebook you'll run the loop in.)

### 3. Notebook settings
- **Accelerator:** GPU T4 ×2 (or P100).
- **Internet:** ON (required for git/pip/HF downloads).
- **Persistence:** not needed — state lives in GitHub.

---

## The bootstrap cell

Paste this as the **only** cell, then **Run All** (or schedule it). It clones the repo
using the secret, sets env vars, and launches the loop.

```python
import os
from kaggle_secrets import UserSecretsClient

# --- config you may edit ---
os.environ["REPO_SLUG"]      = "ml-researcher123/thesis-code"
os.environ["GIT_BRANCH"]     = "research"
os.environ["WORKDIR"]        = "/kaggle/working/repo"
os.environ["POLL_SECONDS"]   = "120"      # how often to check GitHub for new configs
os.environ["RUN_DEVICE"]     = "auto"     # auto -> uses the GPU
os.environ["MAX_MINUTES"]    = "525"      # exit cleanly before Kaggle's ~9h cap
os.environ["GIT_USER_NAME"]  = "kaggle-runner"
os.environ["GIT_USER_EMAIL"] = "kaggle-runner@users.noreply.github.com"
# ---------------------------

os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")

slug   = os.environ["REPO_SLUG"]
branch = os.environ["GIT_BRANCH"]
work   = os.environ["WORKDIR"]
token  = os.environ["GITHUB_TOKEN"]

# clone (or refresh) then hand off to the in-repo loop
if not os.path.isdir(work):
    os.system(f'git clone --branch {branch} '
              f'https://x-access-token:{token}@github.com/{slug}.git {work}')

# install any extra deps the queued experiments need (core is preinstalled on Kaggle)
os.system(f'pip install -q -r {work}/kaggle/requirements.txt')

# run the infinite watch loop (blocks until MAX_MINUTES or session end)
os.system(f'python {work}/kaggle/watch_loop.py')
```

---

## How to drive it (Claude's side)

1. Write/modify an experiment + a `configs/<name>.yaml`.
2. `git push origin research`.
3. Within `POLL_SECONDS`, Kaggle pulls, runs it, and pushes `outputs/<name>/`
   (`results.json`, `summary.md`, figures, `run.log`).
4. Next session: read `outputs/<name>/summary.md`, interpret, update
   `notes/research-log.md`, and queue the next config.

A config is considered **done** when `outputs/<name>/results.json` exists, so the loop
never re-runs finished work. To force a re-run, delete that file (or bump the config
`name:`).

---

## Limits to respect
- **~30 GPU-hours/week** free quota — keep each config's job short and resumable.
- **~9–12h** session wall-clock — `MAX_MINUTES` exits before the kill; just re-run the
  notebook (or use Kaggle's **Schedule** to relaunch daily) to keep the loop alive.
- **Large artifacts** (model weights, indices, cached embeddings) must go to a **Kaggle
  Dataset or Hugging Face Hub**, not git. Only results/figures/logs belong in `outputs/`.
- If a run crashes, the loop pushes a `FAILED.txt` with the traceback so Claude can
  diagnose it next session instead of silently stalling.
