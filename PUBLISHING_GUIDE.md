# Publishing Guide - Python SDK

This guide covers how to publish the ModelSignature Python SDK to PyPI.

## Prerequisites

1. **PyPI Account**: Ensure you have accounts on both [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)
2. **API Tokens**: Generate API tokens for both PyPI and TestPyPI
3. **Build Tools**: Install required build tools

```bash
pip install build twine
```

## Publishing Steps

### 1. Version Update

Update the version in `pyproject.toml`:
```toml
version = "0.2.1"  # Increment appropriately
```

### 2. Update Changelog

Add a new section to `CHANGELOG.md` documenting the changes in this release.

### 3. Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ src/*.egg-info/

# Build source distribution and wheel
python -m build
```

This creates files in the `dist/` directory:
- `modelsignature-0.2.1.tar.gz` (source distribution)
- `modelsignature-0.2.1-py3-none-any.whl` (wheel)

### 4. Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ modelsignature
```

### 5. Publish to PyPI

```bash
# Upload to PyPI
twine upload dist/*
```

### 6. Verify Installation

```bash
# Install from PyPI
pip install modelsignature

# Test basic functionality
python -c "from modelsignature import ModelSignatureClient; print('SDK installed successfully')"
```

## Configuration

### PyPI Configuration

Create `~/.pypirc` for authentication:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHl...  # Your PyPI API token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHl...  # Your TestPyPI API token
```

### Alternative: Environment Variables

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHl...  # Your API token
```

## Release Checklist

- [ ] Version bumped in `pyproject.toml`
- [ ] Changelog updated with new version
- [ ] All tests passing
- [ ] Documentation updated if needed
- [ ] Built package locally (`python -m build`)
- [ ] Tested on TestPyPI
- [ ] Published to PyPI
- [ ] Tagged release in Git
- [ ] Verified installation works

## Git Tagging

After successful PyPI publication:

```bash
git tag v0.2.1
git push origin v0.2.1
```

## Troubleshooting

### Common Issues

1. **Version conflicts**: Ensure version in `pyproject.toml` hasn't been used before
2. **Authentication**: Verify API tokens are correct and have necessary permissions
3. **File upload errors**: Check that all required files are in the `dist/` directory

### Testing Locally

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest

# Type checking
mypy src/modelsignature
```