## ADDED Requirements

### Requirement: Compare Hamming strategies on a loaded dataset
The benchmark tooling SHALL be able to run the same in-memory dataset and query cohort protocol under multiple Hamming trie search strategies and record per-strategy latency and candidate statistics.

#### Scenario: Multi-strategy run
- **WHEN** the benchmark is invoked with a list of strategies including at least `astar`, `multiprobe`, and `beam`
- **THEN** the JSON report SHALL include separate timing summaries keyed by strategy name

### Requirement: Recall against fullscan ground truth
For strategy comparison runs, the benchmark SHALL compute recall@K of each strategy’s neighbor ids against exact fullscan neighbors on the same queries (same K grid).

#### Scenario: Recall field present
- **WHEN** a strategy comparison report is written
- **THEN** each strategy summary SHALL include recall@K (mean over queries) versus fullscan ground truth for each K in the grid

### Requirement: Expansion / probe accounting
The benchmark SHALL record a comparable work metric per strategy (e.g. Bloom probes or node expansions) so latency is not the only cost signal.

#### Scenario: Work metric in JSON
- **WHEN** inspecting a per-query strategy run record
- **THEN** it SHALL include an integer work counter such as `expansions` or `bloom_probes`
