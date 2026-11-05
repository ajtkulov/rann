## ADDED Requirements

### Requirement: Persist benchmark run as JSON
The system SHALL write each completed benchmark run to a JSON file that includes schema version, dataset path, mode, seed, K values, load timing, vector count, per-query cohort/id/word, per-(query,K) elapsed time and neighbor list, and aggregate timing summaries.

#### Scenario: JSON written after run
- **WHEN** a benchmark run finishes successfully
- **THEN** a JSON file SHALL exist at the configured output path and SHALL parse as an object with `version` equal to `1`

#### Scenario: Per-query neighbor payload
- **WHEN** a run for query id `Q` at `K` completes
- **THEN** the JSON SHALL include up to `K` neighbors with id, distance, and word label when vocab is available

### Requirement: Aggregate timing summaries
The JSON SHALL include summary statistics over all queries for each K (at least mean, min, max, and a median/p50) and SHALL break down the same statistics by cohort (`first` vs `random`).

#### Scenario: Summary by K present
- **WHEN** inspecting the `summary.by_k` object for K=10
- **THEN** it SHALL contain numeric fields for mean and p50 elapsed milliseconds computed over all queries run at K=10

#### Scenario: Cohort breakdown present
- **WHEN** inspecting `summary.by_cohort_k`
- **THEN** both `first` and `random` cohorts SHALL have per-K timing aggregates

### Requirement: Configurable output location
The runner SHALL allow the caller to set an output path/stem for the JSON report.

#### Scenario: Custom output stem
- **WHEN** the runner is invoked with an explicit output stem
- **THEN** the JSON report SHALL be written using that stem (with a `.json` suffix)
