## ADDED Requirements

### Requirement: Query projection before search
The system SHALL convert a query float vector into an `H`-bit binary code using the same hyperplanes as the index before traversing the virtual trie.

#### Scenario: Query uses index planes
- **WHEN** a search is requested with a float query vector
- **THEN** the system SHALL hash the query with the index’s stored planes and use that binary code as the A* target

### Requirement: A* traversal minimizing Hamming distance
The system SHALL traverse the virtual trie with A* (or equivalent best-first search) starting at the root, expanding `NN`-bit edge candidates that pass the prefix Bloom filter, and ordering expansion to minimize Hamming distance between the query hash and the hash-bit portion of the path. Item id bits SHALL NOT contribute to the Hamming score.

#### Scenario: Prefer lower Hamming prefixes
- **WHEN** two prefixes of equal depth are candidates and one has smaller Hamming distance to the query hash
- **THEN** the lower-Hamming prefix SHALL be expanded before the higher-Hamming one (ties may break arbitrarily)

#### Scenario: Prefix Bloom gates edges
- **WHEN** an `NN`-bit edge label at the current depth is not claimed by the prefix Bloom filter
- **THEN** that edge SHALL NOT be expanded

#### Scenario: Reach full hash depth
- **WHEN** A* has selected a path of `H` hash bits
- **THEN** the system SHALL proceed to enumerate valid id suffixes for that hash using Bloom probes

### Requirement: Collect valid ids into a buffer
After reaching hash depth, the system SHALL discover item ids whose `hash||id` keys are claimed by the prefix and terminal Bloom filters and SHALL collect those ids into an array buffer.

#### Scenario: Terminal confirmation required
- **WHEN** an id suffix is considered for a completed hash path
- **THEN** the id SHALL be added to the candidate buffer only if the terminal Bloom filter claims the full `hash||id` key

#### Scenario: Candidates returned as buffer
- **WHEN** id enumeration for the search completes
- **THEN** the collected ids SHALL be available as a contiguous array/buffer of integer ids

### Requirement: Top-K rerank by distance function
The system SHALL map candidate ids to their original float vectors, compute a configurable distance to the query (default: cosine distance), sort ascending by distance, and return the top-K neighbors.

#### Scenario: Default cosine ranking
- **WHEN** search is invoked with `K` and the default distance function
- **THEN** results SHALL be the `K` candidates with smallest cosine distance to the query, ordered nearest-first

#### Scenario: K larger than candidate set
- **WHEN** fewer than `K` valid candidates are found
- **THEN** the system SHALL return all found neighbors sorted by distance

#### Scenario: Phantom Bloom ids dropped
- **WHEN** a terminal Bloom hit refers to an id with no stored vector in the index sidecar
- **THEN** that id SHALL be excluded from the ranked results

### Requirement: Expansion budget safety valve
The system SHALL support a configurable maximum A* expansion limit to bound search work on large or noisy Bloom-backed tries.

#### Scenario: Budget exhausted
- **WHEN** the expansion budget is reached before exhaustive exploration
- **THEN** the system SHALL stop expanding new nodes and SHALL rerank whatever candidates were already collected
