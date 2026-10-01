"""Synthetic structured-signal proof; no network or agent framework required."""
import agentassert_abc as aa
from agentassert_abc.integrations.generic import GenericAdapter


def main() -> None:
    contract = aa.loads("""contractspec: "0.1"
kind: agent
name: synthetic-release-check
description: Checks a caller-supplied approval flag, not a security classifier.
version: "1.0.0"
invariants:
  hard:
    - name: release-approved
      check:
        field: release.approved
        equals: true
""")
    adapter = GenericAdapter(contract)
    allowed = adapter.check({"release.approved": True})
    if allowed.hard_violations != 0:
        raise AssertionError("The approved synthetic state must pass")
    print("Allowed state: 0 hard violations")
    try:
        adapter.check_and_raise({"release.approved": False})
    except aa.ContractBreachError:
        print("Unapproved state: ContractBreachError")
    else:
        raise AssertionError("The unapproved synthetic state must raise")


if __name__ == "__main__":
    main()
