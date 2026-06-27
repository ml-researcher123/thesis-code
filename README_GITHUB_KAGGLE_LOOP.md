# GitHub-Controlled Kaggle Loop

This is the stable autonomous setup for running ACE-RAG experiments on Kaggle while keeping a Google Drive mirror for later Colab use.

## Control Flow

```text
Codex updates control/main.py locally and mirrors it to Google Drive.
The same control/main.py is committed to GitHub.
Kaggle runs kaggle_github_loop.py once.
Kaggle polls GitHub for control/main.py changes.
When control/main.py changes, Kaggle runs it.
Kaggle pushes logs and metrics to GitHub under kaggle_results/latest.
Codex reads the GitHub results and prepares the next control/main.py.
```

## Kaggle Inputs

Add these once:

```text
kaggle_github_loop.zip
ace_rag_research_kaggle_ready_v12_routerfix.zip
```

For MuSiQue jobs, also add:

```text
musique.jsonl
```

or the MuSiQue zip.

## Kaggle Secrets

```text
GITHUB_TOKEN = token with contents read/write
GITHUB_REPO = ml-researcher123/thesis-code
GITHUB_BRANCH = main
```

## Kaggle Cell

```python
!python $(find /kaggle/input -name "kaggle_github_loop.py" | head -n 1) --poll-seconds 300
```

## GitHub Repo Layout

The repo should contain:

```text
control/main.py
```

Kaggle will write:

```text
kaggle_results/latest/
kaggle_results/<timestamp>/
```

## Google Drive Mirror

The same control files are mirrored under:

```text
G:\My Drive\my-research\github_control
```

This keeps the current Kaggle/Colab control state available even if GitHub or Kaggle sessions are reset.
