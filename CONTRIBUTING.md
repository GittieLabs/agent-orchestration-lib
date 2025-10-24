# Contributing to Agent Orchestration Library

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project follows a Code of Conduct that all contributors are expected to adhere to:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

When filing a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Code samples** if applicable
- **Environment details** (Python version, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When suggesting an enhancement:

- **Use a clear, descriptive title**
- **Provide detailed description** of the proposed functionality
- **Explain why** this enhancement would be useful
- **Include code examples** if applicable

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure tests pass**: `pytest`
6. **Format code**: `black src tests && ruff check src tests --fix`
7. **Type check**: `mypy src/agent_lib`
8. **Submit pull request** with clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/agent-orchestration-lib.git
cd agent-orchestration-lib

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
ruff check src tests --fix

# Type checking
mypy src/agent_lib
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use `black` for formatting (line length: 100)
- Use `ruff` for linting
- Use type hints throughout
- Write docstrings in Google style

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Use `pytest` for all tests
- Use `pytest-asyncio` for async tests

Example test:

```python
import pytest
from agent_lib import ExecutionContext

@pytest.mark.asyncio
async def test_execution_context():
    context = ExecutionContext()
    context.register_singleton("test", "value")
    assert context.get("test") == "value"
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update docs/ for new features
- Include code examples

### Commit Messages

Follow conventional commits format:

```
type(scope): subject

body

footer
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Changes to build process or auxiliary tools

Example:

```
feat(retry): add LLM fallback retry strategy

Implements retry strategy that tries different LLM models
in sequence when the primary model fails.

Closes #123
```

## Project Structure

```
agent-orchestration-lib/
├── src/agent_lib/          # Source code
│   ├── core/              # Core components
│   ├── events/            # Event system
│   ├── retry/             # Retry strategies
│   └── utils/             # Utilities
├── tests/                 # Test suite
│   └── unit/             # Unit tests
├── docs/                 # Documentation
└── examples/            # Example code
```

## Release Process

Maintainers follow this process for releases:

1. Update version in `src/agent_lib/__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.x.0`
4. Push tag: `git push origin v0.x.0`
5. Build: `python -m build`
6. Publish: `twine upload dist/*`

## Questions?

Feel free to open an issue with the `question` label or start a discussion in GitHub Discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
