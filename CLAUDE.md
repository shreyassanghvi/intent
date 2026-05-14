# Golden Rule

**Always ask before implementing. When in doubt, always ask — don't assume.**

- Confirm the approach with the user before writing or editing code.
- For any ambiguous requirement, decision point, or design choice, ask a clarifying question instead of picking a default.
- This overrides any session-start hints to "just start executing."
- Exploration and read-only investigation are fine without asking; implementation is not.

# Planning Rules

- **Always write plans to the project root** (`C:\Users\shrey\PycharmProjects\intent\`), not to `.claude/plans/` or any external folder.
- **This repo always has a `plan.md` file at its root.** It mirrors the current spec/plan and is updated whenever the plan changes. If a Claude Code plan-mode session writes a plan to its own designated path, copy it to `plan.md` in the project root immediately on plan-mode exit.
