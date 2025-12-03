# Marker Code Style and Conventions

## Code Formatting and Linting
- **Tool**: Ruff (replaces flake8, black, isort, etc.)
- **Pre-commit hooks**: Configured in `.pre-commit-config.yaml`
- **Ruff version**: v0.9.10
- **Auto-fix**: `ruff check --fix .`
- **Formatting**: `ruff format .`

## Python Standards
- **Python Version**: 3.10+
- **Type Hints**: Use Pydantic models for data validation
- **Docstrings**: Follow standard Python conventions
- **Naming Conventions**:
  - Classes: PascalCase (e.g., `PdfConverter`, `MarkdownRenderer`)
  - Functions/Methods: snake_case (e.g., `convert_single`, `build_document`)
  - Constants: UPPER_SNAKE_CASE
  - Private methods: prefix with `_`

## Project Patterns
- **Dependency Injection**: Pass configurations, models, processors to converters
- **Factory Pattern**: `ConfigParser` generates configurations and instances
- **Strategy Pattern**: Different processors for different block types
- **Pipeline Pattern**: Provider → Builders → Processors → Renderer

## Pydantic Usage
- All schema models inherit from Pydantic `BaseModel`
- Use `pydantic-settings` for configuration management
- Type validation enforced at runtime

## Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports from `marker` package root

## Testing Conventions
- Test files in `tests/` directory
- Test file naming: `test_*.py`
- Use pytest markers for custom PDF files: `@pytest.mark.filename('file.pdf')`
- Mock fixtures for consistent testing
