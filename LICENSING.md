> **NOT LEGAL ADVICE.** This document is a plain-language summary of AgentAssert's licensing model. It is not a legal opinion and does not substitute for advice from qualified counsel. Consult an attorney before making licensing decisions for production deployments.

# AgentAssert Licensing

`agentassert-abc` ships under a **dual license**:

| Use case | License |
|---|---|
| Open-source, research, internal deployment, AGPL-compatible products | **AGPL-3.0-or-later** — free |
| Proprietary, closed-source, or hosted SaaS | **Commercial License** — contact Qualixar |

---

## 1. Open License — AGPL-3.0-or-later

Full license text: [LICENSE](LICENSE).

AGPL-3.0 is a strong copyleft license. It grants broad rights to use, modify, and distribute this software at no cost, subject to two primary obligations.

### Network-use copyleft (the AGPL distinction)

AGPL-3.0 Section 13 extends GPL's copyleft to *network-served* software. If you deploy a modified version of AgentAssert and make it accessible to external users over a network — as an API endpoint, a hosted behavioral monitoring service, or a multi-tenant agent platform — you must offer those users access to your modified source code under AGPL, at no charge.

This is the critical difference between AGPL and GPL. It closes the "SaaS loophole": you cannot run modified AGPL software as a hosted service while keeping your modifications proprietary.

### Derivative work copyleft

If you distribute a modified version of AgentAssert, the modified work must also be released under AGPL-3.0-or-later. You cannot re-license modified copies under a proprietary or more permissive license.

### What AGPL does not restrict

- **Internal use** within your organization, with no distribution or network exposure to external users.
- **Research and evaluation** — academic, educational, and prototyping use.
- **Open-source products** built on AgentAssert, provided they are released under AGPL-3.0-or-later or a compatible copyleft license and comply with AGPL's terms.

---

## 2. When You Need a Commercial License

A commercial license removes AGPL's copyleft obligations. You need one if any of the following apply:

1. **Closed-source product.** You are embedding AgentAssert in a proprietary product and do not intend to release that product's source code under AGPL.

2. **Hosted SaaS or API.** You are offering AgentAssert's capabilities as part of a hosted service and do not want to provide network users access to your modified source.

3. **Corporate AGPL policy.** Your legal or compliance team prohibits AGPL dependencies in commercial software distributions.

4. **Contractual certainty.** You need explicit warranty terms, indemnification, or SLA commitments not provided by the open-source license.

If your product is fully open-source and AGPL-compatible, a commercial license is not required.

See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) to request commercial terms.

---

## 3. npm CLI Wrapper — AGPL-3.0-or-later

The npm package `agentassert-abc` at `claim/npm/` is licensed under **AGPL-3.0-or-later**, identical to the Python engine. There is no permissive-license carve-out for the wrapper. JavaScript and TypeScript consumers that depend on this package are therefore subject to AGPL's copyleft obligations — the same obligations that apply to any other consumer of AgentAssert. `claim/npm/package.json` correctly declares `"license": "AGPL-3.0-or-later"`.

If AGPL is incompatible with your project, obtain a commercial license (see Section 4).

---

## 4. Obtaining a Commercial License

Contact Qualixar to discuss terms:

- **Email:** licensing@qualixar.com
- **Website:** https://qualixar.com

Include in your inquiry: a description of your product, how you intend to use AgentAssert, and your expected deployment scale. We respond within five business days.

---

## 5. Contributor Licensing

All external contributions to AgentAssert are accepted subject to a Contributor License Agreement (CLA). By submitting a pull request, contributors grant Varun Pratap Bhardwaj the right to distribute their contributions under both AGPL-3.0-or-later and the Qualixar commercial license. This CLA is a legal prerequisite for the dual-licensing model to remain valid — without it, Qualixar cannot offer code from external contributors under commercial terms.

> By opening a pull request you agree to the CLA described above; see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 6. Copyright

Copyright (c) 2026 Varun Pratap Bhardwaj. All rights reserved.

AgentAssert is part of the [Qualixar](https://qualixar.com) AI Agent Reliability Engineering platform.

---

> **NOT LEGAL ADVICE.** The above is a plain-language summary intended to orient users and integrators. It does not constitute legal counsel. Any binding licensing decision for production use should be reviewed by qualified legal counsel familiar with open-source and commercial software licensing in your jurisdiction.
