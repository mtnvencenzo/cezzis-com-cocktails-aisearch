# 🍸 Contributing to Cezzis Cocktails RAG Agent

Thank you for your interest in contributing to the Cezzis Cocktails AI Search API! This repository provides the FastAPI-based backend for cocktail search and conversational capabilities as part of the broader Cezzis.com RAG and agentic workflow ecosystem. We welcome contributions that help improve API endpoints, vector search, query processing, and integration with the broader Cezzis.com ecosystem.

## 📋 Table of Contents

- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Contributing Process](#-contributing-process)
- [Code Standards](#-code-standards)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Getting Help](#-getting-help)

## 🚀 Getting Started

### 🧰 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.12+
- Make
- Docker & Docker Compose
- Terraform (optional, for IaC under `terraform/`)
- Git


### 🗂️ Project Structure

```text
cezzis-com-cocktails-aisearch/
├── src/
│   └── cezzis_com_cocktails_aisearch/    # FastAPI app, routers, services, vector search
├── test/                                 # Unit and integration tests
├── .github/                              # GitHub workflows and templates
├── Dockerfile, pyproject.toml, README.md # Project config and docs
```


### 🎯 Application Overview

This repository contains the backend API for:
- **Cocktail Search**: Semantic search for cocktails using vector embeddings
- **Conversational AI (WIP)**: Future contextual chat about cocktails via LLMs

## 💻 Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/mtnvencenzo/cezzis-com-cocktails-rag-agent.git
   cd cezzis-com-cocktails-rag-agent
   ```

5. **Docker Compose (Optional)**
   ```bash
   docker compose up
   ```

## 🔄 Contributing Process

### 1. 📝 Before You Start

- **Check for existing issues** to avoid duplicate work
- **Create or comment on an issue** to discuss your proposed changes
- **Wait for approval** from maintainers before starting work (required for this repository)

### 2. 🛠️ Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our [code standards](#-code-standards)

3. **Test your changes**
   ```bash
   make test
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat(extraction): add new functionality for ..."
   ```
   
   Use [conventional commit format](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `style:` for formatting changes
   - `refactor:` for code refactoring
   - `test:` for adding tests
   - `chore:` for maintenance tasks

### 3. 📬 Submitting Changes

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request**
   - Use our [PR template](pull_request_template.md)
   - Fill out all sections completely
   - Link related issues using `Closes #123` or `Fixes #456`
   - Request review from maintainers

## 📏 Code Standards

### 🐍 Python

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write comprehensive docstrings
- Use structured logging
- Avoid global state; prefer dependency injection
- Keep functions small and focused

### 🧪 Code Quality

```bash
# Run tests
make test

# Lint and formatting with ruff
ruff format .
ruff check .
```

### 🌱 Infrastructure (Terraform)

- **Terraform**: Use Terraform best practices
- **Variables**: Define all variables in `variables.tf`
- **Documentation**: Document all resources and modules
- **State**: Never commit `.tfstate` files

## 🧪 Testing

### 🧪 Unit Tests
```bash
make test
```


### 📏 Test Requirements

- **Unit Tests**: All new features must include unit tests
- **E2E Tests**: Critical user flows should have E2E test coverage
- **Coverage**: Maintain minimum 80% code coverage
- **Test Naming**: Use descriptive test names that explain the behavior

## 🆘 Getting Help

### 📡 Communication Channels

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact maintainers directly for sensitive issues

### 📄 Issue Templates

Use our issue chooser:
- https://github.com/mtnvencenzo/cezzis-com-cocktails-rag-agent/issues/new/choose

### ❓ Common Questions


**Q: How do I run the application locally?**
A: Follow the [Development Setup](#-development-setup) section above. Use Poetry and Uvicorn to run the FastAPI app:
```bash
poetry install
poetry run uvicorn src.cezzis_com_cocktails_aisearch.main:app --reload
```


**Q: How do I run tests?**
A: Use `poetry run pytest` in the project root to run all unit and integration tests.


**Q: Which part should I contribute to?**
A: Check the issue description - it should indicate which API endpoint, search feature, or module is affected. If unsure, ask in the issue comments.

**Q: Can I contribute without approval?**
A: No, all contributors must be approved by maintainers before making changes.

**Q: How do I report a security vulnerability?**
A: Please email the maintainers directly rather than creating a public issue.

## 📜 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](../LICENSE)).

---

**Happy Contributing! 🍸**

For any questions about this contributing guide, please open an issue or contact the maintainers.
