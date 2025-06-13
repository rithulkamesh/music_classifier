# Contributing to Music Genre Classifier

Thank you for your interest in contributing to the Music Genre Classifier! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

### Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/[username]/music-genre-classifier.git
   cd music-genre-classifier
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Run the application**
   ```bash
   uv run streamlit run src/app.py
   ```

## 🛠️ Development Workflow

### Making Changes

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**

   ```bash
   # Run the application
   uv run streamlit run src/app.py

   # Test model training (optional)
   uv run python src/train_ultra_high_performance.py --help
   ```

4. **Commit and push**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Provide a clear description of your changes
   - Reference any related issues
   - Include screenshots for UI changes

## 📝 Code Style Guidelines

### Python Code

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular

### Documentation

- Update README.md for significant changes
- Add inline comments for complex logic
- Include examples in docstrings

### Commit Messages

Follow conventional commit format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests

## 🎯 Areas for Contribution

### High Priority

- **Model improvements**: New architectures, better features
- **Performance optimization**: Faster training, inference
- **Documentation**: Tutorials, API docs, examples
- **Testing**: Unit tests, integration tests
- **UI/UX**: Streamlit interface improvements

### Medium Priority

- **Data processing**: Better audio preprocessing
- **Visualization**: Enhanced UMAP plots, new chart types
- **Deployment**: Docker, cloud deployment guides
- **Examples**: Jupyter notebooks, demo scripts

### Low Priority

- **Code cleanup**: Refactoring, optimization
- **CI/CD**: GitHub Actions improvements
- **Internationalization**: Multi-language support

## 🧪 Testing Guidelines

### Before Submitting

- Ensure the Streamlit app runs without errors
- Test audio file upload functionality
- Verify UMAP visualization works
- Check that model prediction is accurate

### Test Data

- Use the provided sample audio files in `data/samples/`
- Test with various audio formats (WAV, MP3, FLAC)
- Verify edge cases (very short/long files, corrupted audio)

## 📋 Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Changes are tested and working
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] PR description is clear and complete
- [ ] No unnecessary files are included

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Error messages and stack traces

### Feature Requests

For new features, please describe:

- The problem you're trying to solve
- Proposed solution
- Alternative approaches considered
- Implementation difficulty estimate

## 🤝 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Acknowledge different skill levels and backgrounds

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Sharing private information without permission

## 📞 Getting Help

### Questions?

- Open an issue with the "question" label
- Check existing issues and documentation first
- Provide context and specific details

### Discussion

- Use GitHub Discussions for broader topics
- Share ideas, use cases, and feedback
- Help other users with their questions

## 🙏 Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes for significant contributions
- Project documentation

Thank you for helping make this project better! 🎵
