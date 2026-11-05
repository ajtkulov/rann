## ADDED Requirements

### Requirement: Pluggable Hamming trie search strategies
The system SHALL provide multiple strategies for discovering low Hamming-distance hash paths on the Bloom-gated virtual trie, selectable at search time, without requiring reindexing.

#### Scenario: Strategy selection at search
- **WHEN** a caller requests ANN search with an explicit strategy identifier
- **THEN** the system SHALL use that strategy for hash-path discovery before id enumeration and distance rerank

#### Scenario: Default remains available
- **WHEN** a caller uses the default search entrypoint without naming a strategy
- **THEN** the system SHALL use the A*/Dijkstra Hamming search strategy (current behavior)

### Requirement: Multi-probe increasing-radius strategy
The system SHALL implement a multi-probe strategy that searches paths within Hamming radius `r = 0, 1, …, r_max` in order, only expanding Bloom-positive edges whose chunk labels keep the path within the current radius budget.

#### Scenario: Exact match radius first
- **WHEN** multi-probe runs with `r_max ≥ 0`
- **THEN** it SHALL attempt to collect candidates along Hamming-0 paths relative to the query hash before expanding to radius 1

#### Scenario: Early stop on enough terminals
- **WHEN** multi-probe has collected at least a configured minimum number of terminal candidates before `r_max`
- **THEN** it MAY stop without exploring higher radii

### Requirement: Beam search strategy
The system SHALL implement a beam-search strategy that, at each trie depth, retains at most `beam_width` prefixes with the smallest Hamming cost so far and expands only those prefixes.

#### Scenario: Bounded fringe
- **WHEN** beam search runs with `beam_width = W`
- **THEN** after each depth’s selection step at most `W` prefixes SHALL remain for further expansion

#### Scenario: Non-exact caveat
- **WHEN** beam search completes under a finite `W`
- **THEN** the system does not claim that all globally minimal Hamming codes were explored

### Requirement: Shared post-processing
All Hamming strategies SHALL feed discovered hash paths into the same id-suffix enumeration (Prefix + Terminal Bloom) and the same pluggable distance rerank pipeline.

#### Scenario: Rerank unchanged
- **WHEN** any strategy returns candidate ids
- **THEN** top-K ordering SHALL still be by the configured vector distance (default cosine)
