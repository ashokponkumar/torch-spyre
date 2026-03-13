---
description: "Trigger and test GitHub Actions workflows to validate workflow changes"
---

# Test Action Changes

When you modify files in `.github/workflows/`, use this skill to automatically trigger workflows and validate your changes before pushing.

## Usage

After modifying a workflow file, use the test-action skill to trigger it:

```
/test-action <workflow_name>
```

Or test all workflows:

```
/test-action all
```

## Available Workflows

- `pytest` — Run pytest test suite
- `linters` — Run pre-commit linting checks
- `arc-demo` — Actions Runner Controller demo

## Commands

- `/test-action pytest` — Trigger pytest workflow
- `/test-action linters` — Trigger linters workflow
- `/test-action arc-demo` — Trigger arc-demo workflow
- `/test-action all` — Trigger all workflows
- `/test-action <name> --wait` — Wait for workflow to complete
- `/test-action <name> --logs` — Show full workflow logs

## Examples

Test the pytest workflow:
```
/test-action pytest
```

Test and wait for results:
```
/test-action pytest --wait
```

Test all workflows before pushing:
```
/test-action all --wait
```

## Notes

- Workflows are triggered via `workflow_dispatch` on the `ashokponkumar-test-org/torch-spyre` test repository
- View recent runs: `gh api repos/ashokponkumar-test-org/torch-spyre/actions/runs --jq '.workflow_runs[] | {name, conclusion, created_at}'`
