## ADDED Requirements

### Requirement: Dual Bloom-filter virtual trie
The system SHALL store indexed items in memory using two Bloom filters: a prefix filter that records trie path prefixes, and a terminal filter that records full `hash_bits || id_bits` keys.

#### Scenario: Insert updates both filters
- **WHEN** an item with binary hash code and item id is inserted
- **THEN** the prefix Bloom filter SHALL contain every `NN`-aligned prefix of the hash (and of `hash||id` for id-bit depths), and the terminal Bloom filter SHALL contain the full `hash||id` bitstring

#### Scenario: Prefix membership probe
- **WHEN** a caller checks whether a given `NN`-bit-aligned prefix may exist
- **THEN** the system SHALL return whether the prefix Bloom filter claims membership

#### Scenario: Terminal membership probe
- **WHEN** a caller checks whether a full `hash||id` key may exist
- **THEN** the system SHALL return whether the terminal Bloom filter claims membership

### Requirement: Chunked edge labels
The system SHALL split binary keys into chunks of `NN` bits (default `NN=6`) for trie edge labels. When the bit length is not divisible by `NN`, the final chunk SHALL be zero-padded to `NN` bits.

#### Scenario: Default chunk size
- **WHEN** an index is created without overriding `NN`
- **THEN** edge labels SHALL be `6` bits wide (`64` possible labels per depth)

#### Scenario: Aligned default H
- **WHEN** `H=132` and `NN=6`
- **THEN** the hash portion SHALL span exactly `22` chunks with no padding

### Requirement: Item id width
The system SHALL support item ids as either 32-bit or 64-bit signed integers, appending the id’s bit representation to the hash bits to form the terminal key.

#### Scenario: Terminal key length with 32-bit ids
- **WHEN** the index is configured for 32-bit ids and `H=132`
- **THEN** each terminal key SHALL be `164` bits before chunk padding

#### Scenario: Terminal key length with 64-bit ids
- **WHEN** the index is configured for 64-bit ids and `H=132`
- **THEN** each terminal key SHALL be `196` bits before chunk padding

### Requirement: No reliable deletion in v1
The system SHALL NOT provide an item-deletion API that guarantees removal from the Bloom filters.

#### Scenario: Delete API absent or documented unsupported
- **WHEN** a caller inspects the public index API
- **THEN** there SHALL be no operation that claims durable deletion of a previously inserted id from the Bloom filters
