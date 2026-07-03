# Colab Auto-Runner / Pusher — Setup

This is the configuration for the autonomous GPU loop, running on **Google Colab** instead
of Kaggle. Once set up, the flow is:

```
Claude (local)  --push code+configs-->  GitHub (branch: research)
                                              |
                                              v
Colab notebook (GPU) --poll--> pull --> run pending configs --> push results to outputs/
                                              |
                                              v
Claude (next session) <--read outputs/-- GitHub   (interprets, writes research-log, queues next config)
```

The runner only ever writes under `outputs/`. Claude only writes code, `configs/`, and
`notes/`. Disjoint paths ⇒ no merge conflicts.

**Do not run this alongside the Kaggle watch loop against the same branch.** Two
independent loops polling/pushing at once can race — the idempotency check via
`results.json` is not atomic across separate machines. Use one or the other.

---

## One-time setup

### 1. Create a GitHub Personal Access Token (PAT)
- GitHub → Settings → Developer settings → **Personal access tokens** → *Fine-grained*.
- Repository access: only `ml-researcher123/thesis-code`.
- Permissions: **Contents: Read and write**.
- Copy the token (starts with `github_pat_...`).

### 2. Store it as a Colab Secret
- Open the notebook → click the **key icon (🔑)** in the left sidebar → **Secrets**.
- **Add new secret**: name `GITHUB_TOKEN`, value the PAT.
- Toggle **"Notebook access"** ON for this secret (off by default — easy to miss).

### 3. Runtime settings
- **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 on the free tier;
  A100/L4 available on Colab Pro/Pro+).
- **Internet:** on by default in Colab (no toggle needed, unlike Kaggle).

### 4. The idle-disconnect gotcha (read this before you walk away)
Kaggle sessions run to their wall-clock cap regardless of whether you're watching.
**Colab's free tier is different: it disconnects the runtime after ~90 minutes of no
browser interaction**, even while this loop is actively running GPU jobs. This is
separate from `MAX_MINUTES` below and can't be fixed from inside the script. Practical
mitigations:
- Keep the browser tab open and occasionally interact with it (click, scroll).
- Colab Pro/Pro+ gives longer, more reliable background execution and fewer
  disconnects (though not a hard guarantee of unlimited background runtime).
- If the runtime disconnects, just reconnect and re-run the bootstrap cell below — the
  loop resumes exactly where it left off (idempotency via `results.json`; nothing is lost).

---

## The bootstrap cell

Paste this as a cell and run it. It clones the repo using the secret, sets env vars, and
launches the loop.

```python
import os
from google.colab import userdata

# --- config you may edit ---
os.environ["REPO_SLUG"]      = "ml-researcher123/thesis-code"
os.environ["GIT_BRANCH"]     = "research"
os.environ["WORKDIR"]        = "/content/repo"
os.environ["POLL_SECONDS"]   = "120"      # how often to check GitHub for new configs
os.environ["RUN_DEVICE"]     = "auto"     # auto -> uses the GPU
os.environ["MAX_MINUTES"]    = "525"      # exit cleanly well before any session cap
os.environ["GIT_USER_NAME"]  = "colab-runner"
os.environ["GIT_USER_EMAIL"] = "colab-runner@users.noreply.github.com"
# ---------------------------

os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")

slug   = os.environ["REPO_SLUG"]
branch = os.environ["GIT_BRANCH"]
work   = os.environ["WORKDIR"]
token  = os.environ["GITHUB_TOKEN"]

# clone (or refresh) then hand off to the in-repo loop
if not os.path.isdir(work):
    os.system(f'git clone --branch {branch} '
              f'https://x-access-token:{token}@github.com/{slug}.git {work}')

# install any extra deps the queued experiments need (torch is preinstalled on Colab;
# sentence-transformers/datasets/peft etc. install lazily per-experiment via ensure_deps,
# but installing them up front here avoids a mid-run pip stall)
os.system(f'pip install -q -r {work}/kaggle/requirements.txt')

# run the infinite watch loop (blocks until MAX_MINUTES, disconnect, or session end)
os.system(f'python {work}/colab/watch_loop.py')
```

---

## How to drive it (Claude's side)

1. Write/modify an experiment + a `configs/<name>.yaml`.
2. `git push origin research`.
3. Within `POLL_SECONDS` (while the Colab runtime is connected), it pulls, runs the
   config, and pushes `outputs/<name>/` (`results.json`, `summary.md`, figures, `run.log`).
4. Next session: read `outputs/<name>/summary.md`, interpret, update
   `notes/research-log.md`, and queue the next config.

A config is considered **done** when `outputs/<name>/results.json` exists, so the loop
never re-runs finished work. To force a re-run, delete that file (or bump the config
`name:`).

---

## Limits to respect
- **Idle disconnect (~90 min)** on the free tier is the practical limit, not a fixed
  weekly GPU-hour quota like Kaggle's — Colab doesn't publish an exact number and it
  varies with usage. Keep each config's job short and resumable regardless.
- **Session wall-clock:** generally up to ~12h before Colab recycles the runtime;
  `MAX_MINUTES` exits cleanly before that. Just re-run the bootstrap cell to resume.
- **Large artifacts** (model weights, indices, cached embeddings) must go to Hugging
  Face Hub, not git. Only results/figures/logs belong in `outputs/`.
- If a run crashes, the loop pushes a `FAILED.txt` with the traceback so Claude can
  diagnose it next session instead of silently stalling.
- **Never run this loop and `kaggle/watch_loop.py` at the same time** against the same
  branch — pick one execution environment per session.
