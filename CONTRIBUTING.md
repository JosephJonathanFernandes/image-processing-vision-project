# Contributing to Image Processing Vision Project

First off, thank you for considering contributing to this project! 🎉

## Code of Conduct

By participating in this project, you are expected to uphold professional and respectful behavior.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Screenshots** if applicable
- **Environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List any alternative solutions** you've considered

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following our coding standards:
   - Write clear, commented code
   - Follow PEP 8 style guidelines for Python
   - Add docstrings to functions and classes
   - Update documentation as needed

3. **Test your changes**:
   - Ensure the application runs without errors
   - Test all affected features
   - Add tests if applicable

4. **Commit your changes**:
   ```bash
   git commit -m "Add amazing feature"
   ```
   - Use present tense ("Add feature" not "Added feature")
   - Use imperative mood ("Move cursor to..." not "Moves cursor to...")
   - Reference issues and pull requests when relevant

5. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**:
   - Provide a clear description of the changes
   - Link to any related issues
   - Include screenshots for UI changes

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/your-username/image-processing-vision-project.git
   cd image-processing-vision-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black formatter standard)
- Use meaningful variable and function names

### Documentation

- Add docstrings to all functions, classes, and modules
- Use inline comments for complex logic
- Update README.md for significant changes

### Commit Messages

Good commit message structure:
```
Short summary (50 chars or less)

More detailed explanation if necessary. Wrap at 72 characters.
Include motivation for the change and contrast with previous behavior.

- Bullet points are okay
- Use hyphen or asterisk for bullets
```

## Project Structure Guidelines

```
image-processing-vision-project/
├── app.py                 # Main application (keep modular)
├── utils/                 # Helper functions (if needed)
├── tests/                 # Unit tests (if applicable)
└── docs/                  # Additional documentation
```

## Testing

- Test all image processing operations
- Verify UI components render correctly
- Check error handling for edge cases
- Test with various image formats and sizes

## Need Help?

- Open an issue with the `question` label
- Check existing issues and pull requests
- Review the project README

## Recognition

Contributors will be acknowledged in the project. Thank you for making this project better! 🚀

---

**Remember**: The goal is to make image processing accessible and educational. Every contribution helps!
