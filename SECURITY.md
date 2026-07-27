# Security Policy

## Supported versions

Security updates are provided for the **latest published release** of
`swarm-seek` on [PyPI](https://pypi.org/project/swarm-seek/) (once published).
Until the first PyPI release, report issues against the `main` branch.

| Version lineage | Security updates |
| --------------- | ---------------- |
| Latest PyPI release | Supported |
| Latest release on PyPI | Security fixes for the current minor line |
| Development (`main`) | Best-effort |
| Older published majors/minors | Not regularly backported |

Check the installed version with:

```bash
python -c "import swarm_seek; print(swarm_seek.__version__)"
```

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately by email to the maintainer:

[travis.j.kessler@gmail.com](mailto:travis.j.kessler@gmail.com)

Include:

- A description of the issue and its impact
- Steps to reproduce or a proof of concept (if available)
- Affected versions / commit if known

You should receive an acknowledgment when the report is reviewed. Fixes are
coordinated with the reporter when practical before a public disclosure or
release notes mention.
