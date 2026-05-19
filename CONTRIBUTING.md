# Contributing to Visual Observability Benchmark

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Describe the bug/feature request clearly
- Include steps to reproduce (for bugs)
- Specify your environment (OS, Python version, GPU)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/Visual_Observability_Benchmark.git
cd Visual_Observability_Benchmark

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Format code
black .
```

## Questions?

Feel free to open an issue for any questions or discussions.
