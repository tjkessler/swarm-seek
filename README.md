# Swarm Seek

Literature-validated Artificial Bee Colony (ABC) optimization for Python.

Swarm Seek implements Original ABC and principal peer-reviewed continuous
variants (GABC, qABC, MABC) with a NumPy core, an ask/tell API, and automated
checks against published benchmark figures.

**Status:** `0.1.0` (beta). Continuous ABC core, variants, ask/tell API, and
literature oracles. Docs: https://swarm-seek.readthedocs.io/

## Install

Requires Python 3.10+.

```bash
pip install swarm-seek
```

Editable install for development:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from swarm_seek import ABC, ContinuousSpace
from swarm_seek.benchmarks import sphere

space = ContinuousSpace([(-5.0, 5.0)] * 5)
colony = ABC(space, variant="original", pop_size=20, limit=100, max_evals=5_000, seed=0)
best = colony.minimize(sphere)
print(best.fitness)
```

Ask/tell loop:

```python
import numpy as np

from swarm_seek import ABC, ContinuousSpace
from swarm_seek.benchmarks import sphere

space = ContinuousSpace([(-5.0, 5.0)] * 5)
colony = ABC(space, variant="gabc", pop_size=20, limit=100, max_evals=5_000, seed=0)

while not colony.converged:
    candidates = colony.ask()
    if candidates.size == 0:
        colony.tell([])
        continue
    colony.tell(np.asarray(sphere(candidates), dtype=np.float64))

print(colony.best.fitness)
```

Runnable notebooks with stated pass criteria live in [`examples/`](examples/).
Full documentation (variants, architecture, API, FAQ) builds from `docs/source/`.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff) for citation metadata.
