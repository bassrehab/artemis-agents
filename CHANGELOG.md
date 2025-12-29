# Changelog

All notable changes to ARTEMIS Agents will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-12-29

### Added
- **Hierarchical Debates**: Sub-debates for complex topics with automatic decomposition
  - `HierarchicalDebate` class for orchestrating nested debates
  - `LLMTopicDecomposer` for intelligent topic breakdown
  - `WeightedAverageAggregator` and other verdict aggregation strategies

- **Real-Time Streaming**: Stream argument generation in real-time
  - `StreamingDebate` with async iterator support
  - `StreamEvent` types for granular progress tracking
  - Callback system for custom event handling

- **Steering Vectors**: Runtime behavior modification
  - `SteeringVector` for controlling formality, aggression, evidence emphasis
  - `SteeringController` for applying vectors to prompts
  - Pre-configured presets (formal, aggressive, evidence-focused)

- **Multimodal Evidence**: Support for images and documents
  - `MultimodalEvidenceExtractor` for chart/image analysis
  - Provider-specific content adapters (OpenAI, Anthropic, Google)
  - `DocumentProcessor` for PDF and text extraction

- **Formal Verification**: Argument validity checking
  - `ArgumentVerifier` with configurable rule sets
  - Causal chain validation
  - Citation quality checking
  - Logical consistency verification

- **Causal Analysis**: Enhanced causal reasoning
  - `CausalGraph` for building argument dependency graphs
  - `CausalAnalyzer` for finding weak links and contradictions
  - `CausalVisualizer` for DOT, Mermaid, and HTML exports
  - `CausalStrategy` for attack/defense planning

### Changed
- Updated benchmarks showing ARTEMIS leadership in decision accuracy (86%)
- Simplified docstrings across codebase for better readability
- Improved code consistency and reduced AI-generated patterns

### Fixed
- Various edge cases in safety monitors
- Improved error handling in model adapters

## [1.0.1] - 2025-12-29

### Fixed
- README logo now renders correctly on PyPI

## [1.0.0] - 2025-12-28

### Added
- **Core ARTEMIS Framework**
  - Hierarchical Argument Generation (H-L-DAG) with strategic, tactical, and operational levels
  - Adaptive Evaluation with Causal Reasoning (L-AE-CR)
  - Jury Scoring Mechanism with multi-perspective evaluation
  - Ethics module for argument ethical alignment

- **Model Providers**
  - OpenAI (GPT-4o, o1-preview, o1-mini)
  - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
  - Google (Gemini 2.0, Gemini 1.5 Pro)
  - DeepSeek (R1 with extended thinking)

- **Safety Monitoring**
  - Sandbagging detection with baseline tracking
  - Deception monitoring with factual consistency checking
  - Logical fallacy detection
  - Behavioral drift tracking
  - Ethics guard for boundary enforcement

- **Framework Integrations**
  - LangChain Tool wrapper
  - LangGraph Debate Node
  - CrewAI Tool integration

- **MCP Server**
  - Full MCP protocol support
  - Session management for multi-turn debates
  - CLI for server startup

- **Documentation**
  - Complete API reference
  - Architecture guide
  - Multiple usage examples

- **Benchmarks**
  - Framework comparison against AutoGen, CrewAI, CAMEL
  - 60 structured debates with LLM-as-judge evaluation
  - Results: competitive with established frameworks, lowest variance

---

## Version History Template

## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality

### Deprecated
- Features that will be removed in future versions

### Removed
- Features removed in this version

### Fixed
- Bug fixes

### Security
- Security-related changes

---

## Links

[Unreleased]: https://github.com/bassrehab/artemis-agents/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bassrehab/artemis-agents/releases/tag/v1.0.0
