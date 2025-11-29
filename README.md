# Facial Vision Agent

[![PyPI](https://img.shields.io/pypi/v/facial-vision-agent.svg)](https://pypi.org/project/facial-vision-agent/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/facial-vision-agent.svg)](https://pypi.org/project/facial-vision-agent/)
[![CI](https://github.com/msalsas/facial-vision-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/msalsas/facial-vision-agent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/msalsas/facial-vision-agent.svg)](https://github.com/msalsas/facial-vision-agent/blob/main/LICENSE)

Facial Vision Agent is an AI-powered library for analyzing facial morphology and hair characteristics from images using vision models and vision-capable LLMs.

This package focuses on visual feature extraction — it identifies facial proportions and prominent features and analyzes hair type, length, density and condition. It is designed to be used as a Python package and integrated into larger systems as an agent that processes image analysis tasks.

## Key features

- Facial feature detection: face shape, proportions, forehead, eyebrows, eyes, nose, mouth, chin and jawline.
- Hair analysis: type (straight/wavy/curly/coily), length, approximate color, density and condition.
- Confidence metrics for each analysis stage.
- Lightweight API designed for automated testing and CI-driven publishing.

## Installation

Developer / editable install:

```bash
python -m pip install -e .[dev]
```

Install from PyPI (when published):

```bash
pip install facial-vision-agent
```

## Quick start

```python
from facial_vision_agent import FacialVisionAgent
from agent_core_framework import AgentTask

agent = FacialVisionAgent(openrouter_api_key="YOUR_API_KEY")

# Full image analysis
task = AgentTask(type="analyze_image", payload={"image_path": "photo.jpg"})
response = agent.process(task)
print(response.data)
```

## Testing

Run the test suite locally:

```bash
python -m pytest
```

## Publishing to TestPyPI (manual)

1. Bump the version in `pyproject.toml` (e.g. to `0.2.0`).
2. Build distributions:

```bash
python -m build
```

3. Upload to TestPyPI (create a TestPyPI API token first):

```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

## Notes on CI

A GitHub Actions workflow (`.github/workflows/publish-testpypi.yml`) is included in this repository. To enable automated publishing to TestPyPI, add your TestPyPI token as a repository secret named `TEST_PYPI_API_TOKEN` and run the workflow manually or by pushing a test tag.

## Contributing

Contributions are welcome. Please open an issue or a pull request. Follow the existing code style, add tests for new behavior, and keep changes focused and small.

## License

MIT — see the `LICENSE` file for details.

