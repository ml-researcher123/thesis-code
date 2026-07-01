# Mermaid diagrams for manual inclusion

Two diagrams I could not render cleanly as vector art from a script are given here as
Mermaid.js source. Paste each into any Mermaid renderer (e.g. mermaid.live, or the VS Code
Mermaid extension), export to PDF/PNG, and drop into `figures/`. The paper does **not**
depend on them — Figure 1 (`figures/fig1_concept.pdf`) already carries the core schematic —
so treat these as optional supplements (e.g. for an appendix or slides).

## D1 — Facet-lens routing (the C4 method)

```mermaid
flowchart LR
    Q["query q"] --> R{"router R(q)\nfacet f"}
    R -->|entities| L1["lens 1\nrank d_total/K"]
    R -->|numerics| L2["lens 2\nrank d_total/K"]
    R -->|relations| L3["lens 3\nrank d_total/K"]
    R -->|negation| L4["lens K\nrank d_total/K"]
    L1 --> S["score s(q,d) = <W_q^f q, W_d^f d>"]
    L2 --> S
    L3 --> S
    L4 --> S
    S --> TOPK["top-k docs (facet-specialized)"]
    subgraph BANK["per-lens doc banks (equal total budget d_total)"]
        L1
        L2
        L3
        L4
    end
```

## D2 — Experimental map (how the blocks build on one another)

```mermaid
flowchart TD
    E1["E1/E1c: retrieval wall\n(free vectors, sign-rank)"] --> E3["E3: compounding\n(independent composition)"]
    E2["E2/E5b: compression wall\n(slot-memory / real encoder)"] --> E3
    E1 --> E4["E4b-d: real-model wall\n(arctic, FiQA, PCA)"]
    E1 --> E8["E8: LIMIT adversarial\n(3 embedders, 50k docs)"]
    E3 --> E3b["E3b: dependence\n(Gaussian copula, rho)"]
    E3b --> E7["E7: measured rho ~ 0\n(real pipeline, mxbai)"]
    E3 --> E6["E6: real end-to-end\nallocation (C3)"]
    E1 --> C4["C4: facet-lens escape\n(free vectors)"]
    C4 --> C4b["C4b: real-encoder\nscope (negative)"]
    classDef wall fill:#eef,stroke:#333;
    classDef head fill:#fee,stroke:#333;
    class E1,E2,E4,E8 wall;
    class E3,E7 head;
```
