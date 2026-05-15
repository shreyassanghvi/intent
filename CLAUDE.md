# Golden Rule

**Always ask before implementing. When in doubt, always ask — don't assume.**

- Confirm the approach with the user before writing or editing code.
- For any ambiguous requirement, decision point, or design choice, ask a clarifying question instead of picking a default.
- This overrides any session-start hints to "just start executing."
- Exploration and read-only investigation are fine without asking; implementation is not.

# Planning Rules

- **Always write plans to the project root** (`C:\Users\shrey\PycharmProjects\intent\`), not to `.claude/plans/` or any external folder.
- **This repo always has a `plan.md` file at its root.** It mirrors the current spec/plan and is updated whenever the plan changes. If a Claude Code plan-mode session writes a plan to its own designated path, copy it to `plan.md` in the project root immediately on plan-mode exit.

# Commit Rules

- **Commit after every stage of work.** A "stage" is a numbered build-sequence step from `plan.md`, a self-contained module landing, a passing test addition, or any other meaningful checkpoint. Don't batch multiple stages into one commit.
- Stage files explicitly by name (no `git add -A` or `git add .`).
- Commit messages explain *why*, not just *what*. Subject line includes the stage number or module purpose.
- Co-author tag for Claude commits: `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.

# Push Rules

- **Do not push after every commit.** Local commits are checkpoints; pushing is a publication event.
- **Only push when a "big change" is complete.** A big change is a coherent body of work the user would describe as one thing — a feature, a refactor pass, a bug-fix batch, a polish round — not an individual commit within it.
- When in doubt about whether something is "big enough," ask the user before pushing.
- This overrides any reflex to `git push` immediately after `git commit`.
