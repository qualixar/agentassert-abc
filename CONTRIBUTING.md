# Contributing to AgentAssert

Thank you for your interest in contributing to AgentAssert. This guide covers everything you need to get started.

## Development Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Clone the repo
git clone https://github.com/qualixar/agentassert-abc.git
cd agentassert-abc

# Install dependencies (including dev tools)
uv sync --group dev

# Verify everything works
uv run pytest
uv run ruff check src/ tests/
uv run basedpyright
```

## Development Workflow

1. **Fork** the repository and create a feature branch
2. **Write tests first** (TDD: RED -> GREEN -> REFACTOR)
3. **Implement** your changes
4. **Run the full check suite:**
   ```bash
   uv run ruff check src/ tests/    # Lint
   uv run basedpyright               # Type check
   uv run pytest                     # Tests (309+ must pass)
   ```
5. **Submit a PR** against `main`

## Code Standards

- **Formatting:** Ruff (auto-format with `uv run ruff format src/ tests/`)
- **Type checking:** basedpyright in strict mode -- all public APIs must be fully typed
- **Immutability:** Pydantic models use `frozen=True`. Create new objects, never mutate.
- **File size:** 800 lines max per file, 50 lines max per function
- **Error handling:** Handle errors explicitly. Never `except: pass`.

## Contributing Domain Contracts

We welcome new domain contracts. To add one:

1. Create a YAML file in `contracts/examples/<domain>.yaml`
2. Follow the ContractSpec v0.1 schema (see existing contracts for reference)
3. Add benchmark scenarios in `benchmarks/scenarios/<domain>/`
4. Add a test in `tests/test_integrations/test_domain_contracts.py`
5. Update the contracts table in `README.md`

## Mathematical Formulas

All mathematical formulas are defined in the research paper ([arXiv:2602.22302](https://arxiv.org/abs/2602.22302)). If your contribution involves mathematical components:

- Reference the paper for the canonical formula
- Never approximate or simplify formulas without maintainer approval
- Add property-based tests (hypothesis) for mathematical functions

## Attribution

All source files include a copyright header:

```python
# Copyright 2026 Varun Pratap Bhardwaj & Qualixar
# Licensed under Elastic License 2.0 — see LICENSE
# AgentAssert: Formal Behavioral Contracts for AI Agents
# Paper: arXiv:2602.22302 | https://agentassert.com
```

Add this header to any new `.py` files you create.

## License

By contributing, you agree that your contributions will be licensed under the [Elastic License 2.0](LICENSE).

---

**Questions?** Open a [Discussion](https://github.com/qualixar/agentassert-abc/discussions) or email varun@qualixar.com.
