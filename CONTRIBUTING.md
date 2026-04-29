# Contributing to EvolAI

Thank you for your interest in contributing to EvolAI!

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/evolai-subnet/evolai.git
cd evolai
```

2. Install in development mode:
```bash
pip install -e .
```

3. Install development dependencies:
```bash
pip install pytest black mypy
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Format code with `black`:
  ```bash
  black evolai/
  ```

## Testing

Run tests before submitting:
```bash
pytest tests/
```

## Submitting Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git commit -m "Add: your feature description"
   ```

3. Push and create a Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Areas for Contribution

- **Validator Implementation**: Help build the evaluation loop
- **Miner CLI**: Implement miner registration and serving
- **Testing**: Add unit and integration tests
- **Documentation**: Improve guides and examples
- **Bug Fixes**: Report and fix issues

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on collaboration

## Questions?

Open an issue for discussion before starting major changes.
