## ADDED Requirements

### Requirement: HTML visual report from JSON
The system SHALL generate a self-contained HTML report alongside the JSON that visualizes latency versus K (overall and by cohort) and presents summary tables without requiring network access to render core charts.

#### Scenario: HTML emitted with JSON
- **WHEN** a benchmark run completes with default reporting enabled
- **THEN** an `.html` file SHALL be written next to the `.json` results and SHALL open as a static page showing timing charts or equivalent bar visuals and a summary table

#### Scenario: Offline rendering
- **WHEN** the HTML file is opened without network connectivity
- **THEN** the primary latency-by-K visualization and summary tables SHALL still be visible (no hard dependency on an external CDN for the core view)

### Requirement: Visual cohort comparison
The visual report SHALL distinguish the `first` and `random` query cohorts when showing latency statistics by K.

#### Scenario: Cohort series visible
- **WHEN** a user views the HTML report for a completed run
- **THEN** they SHALL be able to compare first-cohort vs random-cohort latency at each K

### Requirement: Canvas presentation of results
Benchmark results SHALL be presentable in a Cursor canvas that reads the run JSON (or embedded summary) to show latency-by-K charts and key tables for interactive review in the IDE.

#### Scenario: Canvas shows run summary
- **WHEN** a canvas is created or updated from a benchmark JSON run
- **THEN** it SHALL display at least mean latency by K and a compact results table derived from that run
