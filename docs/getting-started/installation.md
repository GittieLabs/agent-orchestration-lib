# Installation

## Requirements

- Python 3.10 or higher
- pip or poetry

## Install from PyPI

```bash
pip install agent-orchestration-lib
```

## Install with Optional Dependencies

### Development Dependencies

```bash
pip install agent-orchestration-lib[dev]
```

Includes: pytest, mypy, black, ruff, mkdocs

### Webhook Support

```bash
pip install agent-orchestration-lib[webhooks]
```

Includes: httpx for HTTP webhook notifications

### All Optional Dependencies

```bash
pip install agent-orchestration-lib[all]
```

## Install from Source

```bash
git clone https://github.com/GittieLabs/agent-orchestration-lib.git
cd agent-orchestration-lib
pip install -e .
```

## Verify Installation

```python
from agent_lib import ExecutionContext, AgentBlock, EventEmitter
print("Installation successful!")
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [Core Concepts](concepts.md)
