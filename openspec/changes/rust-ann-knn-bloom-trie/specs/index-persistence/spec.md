## ADDED Requirements

### Requirement: Serialize Bloom filters and index metadata
The system SHALL serialize an index for an input dataset to a versioned binary blob that includes format version, hyperparameters (`H`, `NN`, id width, dimension `D`, seed), hyperplane coefficients, prefix Bloom filter state, terminal Bloom filter state, and enough information to recover id→vector mappings used for reranking.

#### Scenario: Save after build
- **WHEN** an index has been built from a dataset and serialize is invoked
- **THEN** the system SHALL write a versioned blob containing metadata, planes, both Bloom filters, and vector sidecar data (embedded or clearly referenced)

#### Scenario: Version header present
- **WHEN** a serialized blob is inspected at offset zero
- **THEN** it SHALL begin with a magic/version header that load can validate

### Requirement: Deserialize into a searchable index
The system SHALL load a previously serialized blob into an in-memory index that supports the same search API as a freshly built index.

#### Scenario: Round-trip search equivalence
- **WHEN** an index is serialized and then deserialized
- **THEN** searching the loaded index with the same query and `K` SHALL return the same ordered neighbor ids as the original index (for a deterministic test dataset)

#### Scenario: Reject incompatible version
- **WHEN** load is given a blob with an unsupported format version
- **THEN** the system SHALL return an error and SHALL NOT partial-construct a searchable index

### Requirement: Dataset-oriented build then persist
The system SHALL provide a workflow to build the dual-Bloom index from a dataset of `(id, vector)` pairs and then serialize that index for later reuse.

#### Scenario: Build from dataset then save
- **WHEN** a caller supplies a dataset of vectors with ids, builds the index, and requests serialization
- **THEN** the resulting blob SHALL be reloadable without requiring the original dataset file to be re-hashed from scratch
