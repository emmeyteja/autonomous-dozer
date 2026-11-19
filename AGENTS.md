# Development guidance

Apply the repository-wide architecture requirements in [.cursor/rules/architecture.mdc](.cursor/rules/architecture.mdc) to all development work.

Prioritize modularity, scalability, and reuse of compatible Autoware interfaces. The project brief is a proposal; distinguish implemented behavior from planned capabilities.

## Development flow

- Track the roadmap using GitHub milestones and detailed issues, including dependencies and acceptance criteria.
- Work one issue at a time; record the active task and verification evidence on its issue.
- Commit completed, tested work directly to `main` and push it. Do not create pull requests unless the user changes this instruction.
- Close issues and milestones only when their acceptance criteria are satisfied. Keep planned capabilities distinct from implemented ones.
