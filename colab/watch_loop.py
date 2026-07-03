"""Colab auto-runner / pusher — the infinite watch loop.

This runs *inside a Google Colab notebook*. It:
  1. Pulls the latest `research` branch from GitHub.
  2. Finds every config in `configs/*.yaml` that does NOT yet have
     `outputs/<name>/results.json` (i.e. unprocessed work queued by Claude).
  3. Runs each pending config via `kaggle/run.py` on the Colab GPU (the runner itself is
     platform-agnostic — it just needs a repo checkout and a device — so it lives under
     `kaggle/` and is reused as-is; only this watch loop is Colab-specific).
  4. Commits the produced `outputs/<name>/` and pushes back to GitHub.
  5. Sleeps `POLL_SECONDS` and repeats — so new commits from Claude get picked up
     automatically until the Colab session disconnects or hits `MAX_MINUTES`.

Separation of duties (important):
  - The runner only writes raw artifacts under `outputs/`. It NEVER edits code,
    configs, or `notes/research-log.md`. Claude owns interpretation; the loop owns
    execution. This keeps git conflicts essentially impossible (disjoint paths).

Colab-specific caveat (read this): unlike Kaggle, Colab's **free tier disconnects the
runtime after ~90 minutes of no browser interaction**, even while code is actively running
— this is unrelated to `MAX_MINUTES` below and cannot be worked around from inside this
script. Keep the browser tab open/focused, or use Colab Pro/Pro+ for longer, more reliable
background execution. Do NOT run this loop and the Kaggle watch loop at the same time
against the same branch — two independent loops polling/pushing concurrently can race
(the idempotency check via `results.json` is not atomic across processes).

Configuration is via environment variables (set them in the Colab bootstrap cell,
pulling the token from Colab's Secrets manager — the key icon in the left sidebar):
  GITHUB_TOKEN     (required)  a PAT with `repo` scope for pushing
  REPO_SLUG        (required)  e.g. "ml-researcher123/thesis-code"
  GIT_BRANCH       default "research"
  GIT_USER_NAME    default "colab-runner"
  GIT_USER_EMAIL   default "colab-runner@users.noreply.github.com"
  WORKDIR          default "/content/repo"
  POLL_SECONDS     default "120"
  RUN_DEVICE       default "auto"  (auto|cpu|cuda)
  MAX_MINUTES      default "525"   (~8.75h; stop cleanly before Colab kills us)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import glob


def env(key: str, default: str | None = None) -> str:
    v = os.environ.get(key, default)
    if v is None:
        raise SystemExit(f"[watch] missing required env var: {key}")
    return v


def sh(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[watch] $ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if res.stdout.strip():
        print(res.stdout, flush=True)
    if res.stderr.strip():
        print(res.stderr, flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}")
    return res


def authed_url(slug: str, token: str) -> str:
    return f"https://x-access-token:{token}@github.com/{slug}.git"


def ensure_repo(workdir: str, slug: str, token: str, branch: str) -> None:
    url = authed_url(slug, token)
    if not os.path.isdir(os.path.join(workdir, ".git")):
        os.makedirs(os.path.dirname(workdir), exist_ok=True)
        sh(["git", "clone", "--branch", branch, url, workdir])
    else:
        sh(["git", "remote", "set-url", "origin", url], cwd=workdir)
    sh(["git", "config", "user.name", env("GIT_USER_NAME", "colab-runner")], cwd=workdir)
    sh(["git", "config", "user.email", env("GIT_USER_EMAIL", "colab-runner@users.noreply.github.com")], cwd=workdir)
    sh(["git", "checkout", branch], cwd=workdir)


def sync(workdir: str, branch: str) -> None:
    sh(["git", "fetch", "origin", branch], cwd=workdir)
    sh(["git", "reset", "--hard", f"origin/{branch}"], cwd=workdir)


def pending_configs(workdir: str) -> list[str]:
    out = []
    for cfg in sorted(glob.glob(os.path.join(workdir, "configs", "*.yaml"))):
        name = os.path.splitext(os.path.basename(cfg))[0]
        # config `name:` may differ from filename, but our convention keeps them equal.
        marker = os.path.join(workdir, "outputs", name, "results.json")
        if not os.path.exists(marker):
            out.append(cfg)
    return out


def commit_and_push(workdir: str, branch: str, name: str) -> None:
    sh(["git", "add", "outputs/"], cwd=workdir)
    status = sh(["git", "status", "--porcelain"], cwd=workdir, check=False)
    if not status.stdout.strip():
        print("[watch] nothing to commit", flush=True)
        return
    sh(["git", "commit", "-m", f"results: {name} [colab-runner]"], cwd=workdir)
    # rebase onto any new code/config commits from Claude before pushing
    sh(["git", "pull", "--rebase", "origin", branch], cwd=workdir, check=False)
    sh(["git", "push", "origin", branch], cwd=workdir)


def main() -> int:
    token = env("GITHUB_TOKEN")
    slug = env("REPO_SLUG")
    branch = env("GIT_BRANCH", "research")
    workdir = env("WORKDIR", "/content/repo")
    poll = int(env("POLL_SECONDS", "120"))
    device = env("RUN_DEVICE", "auto")
    max_minutes = float(env("MAX_MINUTES", "525"))

    ensure_repo(workdir, slug, token, branch)
    t_start = time.time()
    print(f"[watch] online. branch={branch} poll={poll}s device={device} budget={max_minutes}min", flush=True)

    while True:
        if (time.time() - t_start) / 60.0 > max_minutes:
            print("[watch] reached MAX_MINUTES budget — exiting cleanly.", flush=True)
            return 0
        try:
            sync(workdir, branch)
            todo = pending_configs(workdir)
            if not todo:
                print(f"[watch] no pending configs; sleeping {poll}s", flush=True)
            for cfg in todo:
                name = os.path.splitext(os.path.basename(cfg))[0]
                print(f"[watch] RUN {name}", flush=True)
                rc = subprocess.run(
                    [sys.executable, os.path.join(workdir, "kaggle", "run.py"),
                     "--config", cfg, "--device", device],
                    cwd=workdir, text=True,
                ).returncode
                # push whatever was produced (results or a FAILED.txt) so Claude sees it
                commit_and_push(workdir, branch, name + ("" if rc == 0 else " (FAILED)"))
        except Exception as exc:  # noqa: BLE001
            print(f"[watch] loop error (continuing): {exc}", flush=True)
        time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())
