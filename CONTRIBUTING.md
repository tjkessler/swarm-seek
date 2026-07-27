# Contributing to Swarm Seek

Thank you for contributing to **Swarm Seek**, a literature-validated Artificial
Bee Colony library for Python. Bug reports, documentation improvements, tests,
and carefully cited variant implementations are welcome.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) (v2.1).
Report unacceptable behavior to
[travis.j.kessler@gmail.com](mailto:travis.j.kessler@gmail.com).

## Development environment

| Requirement | Version |
|-------------|---------|
| Python | 3.10, 3.11, or 3.12 |

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
pip install -e ".[docs]"    # optional, for Sphinx
pre-commit install          # optional, recommended
```

## Running checks locally

```bash
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -m "not slow" --cov=swarm_seek --cov-report=term-missing --cov-fail-under=90
sphinx-build -b html -W --keep-going docs/source docs/build/html
pytest --nbmake examples/ -q   # optional; also run in CI
```

Coverage must stay at or above **90%** on `swarm_seek`. The `-m "not slow"`
flag matches CI and skips full-paper literature oracle runs (use
``pytest tests/ -m slow`` when you need those).

## Adding an ABC variant

New variant modules are welcome when they meet all of the following:

1. **Primary citation** — the introducing paper (and DOI when available) in the
   module docstring, with equation locators for the implemented update rules.
2. **Representation fit** — compatible with the space / backend contracts (or
   extend those contracts with tests and docs).
3. **Validation test** — a literature-oracle fixture from a published table when
   possible, otherwise a synthetic/unit oracle plus an explicit documentation
   note that a paper table is still outstanding.
4. **Registry + docs** — register the variant name, add Sphinx notes, and update
   `CHANGELOG.md`.

PRs that add undocumented “inspired by” heuristics without citations will be
asked to revise.

## Pull request process

1. Open an issue first for substantial API or algorithm changes.
2. Keep the PR focused on one concern.
3. Ensure the local checks above pass.
4. Fill out the PR template checklist (tests, docs, changelog).

## Documentation hosting

Sphinx sources live under `docs/source/` with `.readthedocs.yaml` for
[Read the Docs](https://readthedocs.org/). Until a Read the Docs project is
imported for this repository, build docs locally with the Sphinx command above.
After import, enable the GitHub integration so pushes rebuild
`https://swarm-seek.readthedocs.io/`.

## Security

Report vulnerabilities privately per [SECURITY.md](SECURITY.md). Do not open a
public issue for security reports.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0 (see [LICENSE](LICENSE)).
