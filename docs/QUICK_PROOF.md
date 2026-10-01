# A small runtime contract proof

Install Python 3.12+ and `python -m pip install "agentassert-abc[yaml,math]"`.
From a repository clone, run `python examples/00_quick_proof.py`.
The example contains its YAML contract; it does not depend on a relative contract file.

Expected output:

```text
Allowed state: 0 hard violations
Unapproved state: ContractBreachError
```

Both branches were executed against source version 0.7.1 on 1 October 2026.
The caller supplies the boolean signal. `check` returns violations;
`check_and_raise` raises for the false signal. The application must call the
check before the protected action. This is a synthetic behavior proof, not an
accuracy, latency, PII detection or security guarantee.

See [enforcement coverage](enforcement-coverage.md) for adapter and host boundaries.
