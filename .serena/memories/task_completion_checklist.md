# Task Completion Checklist

## When a Task is Completed

### 1. Code Quality Checks
```bash
# Run Ruff linting with auto-fix
ruff check --fix .

# Run Ruff formatting
ruff format .

# Or run all pre-commit hooks
pre-commit run --all-files
```

### 2. Testing
```bash
# Run relevant tests
pytest tests/path/to/test_file.py

# Or run all tests
pytest

# For integration tests with specific PDFs
pytest -m "filename('test_file.pdf')"
```

### 3. Verification
- Ensure all tests pass
- Verify no Ruff warnings remain
- Check that changes don't break existing functionality
- Test with sample documents if modifying conversion logic

### 4. Documentation
- Update docstrings if adding new functions/classes
- Update CLAUDE.md if changing core architecture or workflows
- Update README.md if adding new features or changing usage

### 5. Git Workflow
```bash
# Check status
git status

# Stage changes
git add .

# Commit with meaningful message
git commit -m "descriptive message"

# Push if working on feature branch
git push origin branch-name
```

## Pre-Commit Hook Behavior
The `.pre-commit-config.yaml` automatically runs:
1. Ruff linting with auto-fix
2. Ruff formatting

These will run on every commit, so manual pre-commit execution is optional but recommended before committing.
