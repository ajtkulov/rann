## ADDED Requirements

### Requirement: Deterministic hyperplane projection
The system SHALL project a multi-dimensional float vector into an `H`-bit binary code by comparing the vector against `H` random hyperplanes generated from a configurable seed (default seed `1729`, default `H=132`).

#### Scenario: Bit set by half-space
- **WHEN** a vector `x` is projected and the dot product with plane `i` is greater than or equal to zero
- **THEN** bit `i` of the binary code SHALL be `1`

#### Scenario: Bit cleared by opposite half-space
- **WHEN** a vector `x` is projected and the dot product with plane `i` is less than zero
- **THEN** bit `i` of the binary code SHALL be `0`

#### Scenario: Same seed yields identical planes
- **WHEN** two projectors are constructed with the same seed, dimension `D`, and `H`
- **THEN** they SHALL produce identical hyperplanes and identical codes for the same input vector

### Requirement: Configurable hash length
The system SHALL allow `H` to be configured as a hyperparameter and SHALL produce a binary code of exactly `H` bits.

#### Scenario: Default hash length
- **WHEN** a projector is created without overriding `H`
- **THEN** the binary code length SHALL be `132` bits

### Requirement: Planes available for persistence
The system SHALL expose the generated hyperplane coefficients so they can be stored with an index and reused at search time without relying on cross-platform RNG replay alone.

#### Scenario: Planes readable after construction
- **WHEN** a projector has been constructed
- **THEN** callers SHALL be able to read all `H` plane coefficient vectors of length `D`
