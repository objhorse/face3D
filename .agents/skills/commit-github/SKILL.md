---
allowed-tools: Bash(git:*), Bash(gh:*)
argument-hint: "[https://github.com/OWNER/REPO.git | OWNER/REPO] [--public]"
description: Commit if needed, push to a specified GitHub remote, set repo private via gh when possible
disable-model-invocation: true
---

## Context

- Git remotes: !`git remote -v`
- Status: !`git status -sb`
- Branch: !`git branch --show-current`
- gh available: !`gh --version 2>/dev/null || echo "gh not installed"`

## Arguments ($ARGUMENTS)

Parse in order:

1. **Remote target** (required unless `origin` already points to the intended GitHub repo):
   - Full HTTPS URL: `https://github.com/OWNER/REPO.git`
   - Or shorthand for `gh`: `OWNER/REPO`
2. **Visibility** (optional):
   - Default: **private** (`--visibility private` when using `gh`)
   - If user passes **`--public`**, skip making the repo private.

## Your task

1. **Ensure there is something to push**  
   If there are unstaged/uncommitted changes and the user clearly wants them included, create a sensible commit (same message style as the `commit` skill: conventional commits). If nothing to commit, continue.

2. **Configure `origin`** (only when the user provided a URL or `OWNER/REPO` in `$ARGUMENTS`)  
   - If a full `https://github.com/...git` URL was given:
     - `git remote remove origin` only if replacing an unwanted origin (confirm intent from context), then `git remote add origin <URL>` **or** `git remote set-url origin <URL>`.
   - If `OWNER/REPO` shorthand was given, treat it as the GitHub repository slug for `gh` and keep using the existing remote if it already matches; otherwise set `origin` to `https://github.com/OWNER/REPO.git`.

3. **Push**  
   - `git push -u origin HEAD` (or `git push -u origin <branch>`).  
   - If the remote rejects because the repo does not exist yet and `gh` is available, create it first, e.g. `gh repo create OWNER/REPO --private --source=. --remote=origin --push` (adjust flags if `--public` was requested).

4. **Set repository to private (GitHub-side)**  
   - Git itself **cannot** set public/private; use **`gh`** when installed and authenticated:
     - `gh repo edit OWNER/REPO --visibility private`  
     - Skip this step if `$ARGUMENTS` contains `--public`.
   - If `gh` is missing or not logged in, tell the user to open **GitHub → Repo → Settings → Danger zone → Change visibility → Private**, and to install/login `gh` for automation next time.

5. **Safety**  
   - Never force-push (`--force`) unless the user explicitly asks.  
   - Do not overwrite remotes blindly; prefer `git remote set-url origin <new>` when updating URL.

---
**Last Updated**: April 2026
