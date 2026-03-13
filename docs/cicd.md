# CI/CD Workflows

This document describes the GitHub Actions workflows configured for torch-spyre.

## Test Workflow

**File:** `.github/workflows/tests.yaml`

Runs pytest tests on the torch-spyre codebase.

- **Trigger:** Push to `main` or `cicd-experiments` branches, or PR targeting these branches
- **Runner:** ashok-aiu-x86-64-arc-runner-set (self-hosted)
- **Matrix:** Python 3.11, 3.12
- **Steps:**
  1. Checkout code
  2. Setup Python
  3. Install uv package manager
  4. Create virtual environment
  5. Install test and build dependencies
  6. Run pytest

## ARC Demo Workflow

**File:** `.github/workflows/arc-demo.yaml`

Demonstrates Actions Runner Controller (ARC) with runner scale sets.

- **Trigger:** Manual dispatch or PR to `cicd-experiments` branch
- **Runner:** ashok-aiu-x86-64-arc-runner-set (self-hosted)
- **Steps:** Simple echo statement demonstrating ARC runner usage

## Linters Workflow

**File:** `.github/workflows/linters.yaml`

Runs pre-commit checks for code quality and linting.

- **Trigger:** Push to `main` or tags, PR to `main`
- **Runner:** ubuntu-latest
- **Steps:**
  1. Checkout code
  2. Setup Python
  3. Install uv
  4. Setup pre-commit
  5. Run pre-commit hooks

## Dependabot Workflow

**File:** `.github/workflows/dependabot.yaml`

Handles automated dependency updates from Dependabot PRs.

- **Trigger:** PR from dependabot[bot]
- **Runner:** ubuntu-latest
- **Steps:**
  1. Recompile requirements using `tools/update-requirements.sh`
  2. Commit and push changes if needed
