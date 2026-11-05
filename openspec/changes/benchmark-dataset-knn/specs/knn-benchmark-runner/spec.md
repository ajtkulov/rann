## ADDED Requirements

### Requirement: Load dataset fully into memory
The benchmark runner SHALL load a serialized dataset index (and optional vocab) from a path prefix into memory before any timed queries.

#### Scenario: Load glove prefix
- **WHEN** the runner is invoked with prefix `data/glove.6B.300d`
- **THEN** it SHALL load the corresponding `.rann` index into memory and record load duration separately from query timings

### Requirement: Fixed plus random query cohorts
The runner SHALL construct a query set consisting of the first 20 vectors (ids `0..19` when present) and 20 additional vectors chosen uniformly at random with a configurable seed, preferring ids outside the first-20 set when the dataset is large enough.

#### Scenario: Default cohorts on large dataset
- **WHEN** the dataset contains at least 40 vectors and the runner uses the default cohort sizes
- **THEN** the query set SHALL contain 20 `first` cohort queries and 20 `random` cohort queries with no overlapping ids

#### Scenario: Seeded randomness is reproducible
- **WHEN** the runner is executed twice with the same seed and same dataset
- **THEN** the random cohort ids SHALL be identical

### Requirement: Timed top-K sweep
For each query in the set, the runner SHALL execute KNN search for each `K` in `{3, 5, 10, 20, 50}` (unless overridden) and SHALL measure wall-clock elapsed time for each `(query, K)` independently.

#### Scenario: Default K grid
- **WHEN** the runner runs with default K values
- **THEN** each query SHALL be searched at K = 3, 5, 10, 20, and 50

#### Scenario: Timing excludes load
- **WHEN** a query search completes
- **THEN** the recorded elapsed time SHALL exclude dataset load time

### Requirement: Search mode selection
The runner SHALL support an exact in-memory cosine KNN mode as the default and MAY support an ANN mode selectable via flag.

#### Scenario: Default exact mode
- **WHEN** the runner is invoked without an explicit mode
- **THEN** searches SHALL use exact cosine top-K over loaded vectors
