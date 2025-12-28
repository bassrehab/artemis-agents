# Benchmark Analysis

**Date:** December 2025
**Author:** Subhadip Mitra

## The Setup

We ran 60 structured debates across four multi-agent frameworks: ARTEMIS, AutoGen, CrewAI, and CAMEL. Each framework debated the same five topics, three times each. An LLM (GPT-4o) scored every debate on argument quality, decision accuracy, and reasoning depth.

The goal was simple: see how ARTEMIS stacks up against established frameworks.

## The Results

| Framework | Argument Quality | Decision Accuracy | Reasoning Depth |
|-----------|------------------|-------------------|-----------------|
| CrewAI | 89.3% | 81.3% | 86.3% |
| **ARTEMIS** | 81.3% | 67.3% | 84.0% |
| AutoGen | 77.2% | 75.0% | 76.4% |
| CAMEL | 71.0% | 45.2% | 73.7% |

CrewAI won. ARTEMIS came second. That's the honest summary.

## What Went Well

**Consistency.** ARTEMIS had the lowest variance across runs. When you run the same debate three times, you get similar scores. CrewAI's scores bounced around more (±4-7 points vs ARTEMIS's ±1-2). For production systems where predictability matters, this is actually meaningful.

**Reasoning depth.** ARTEMIS scored 84% here, close to CrewAI's 86%. The H-L-DAG structure (strategic → tactical → operational arguments) seems to help produce more layered reasoning. This was the hypothesis going in, and the data supports it.

**It works.** All 15 ARTEMIS debates completed without errors. The jury deliberated, verdicts were reached, and the whole pipeline held together. That's table stakes, but worth noting for a v1.

## What Didn't

**Decision accuracy lagged.** 67.3% vs CrewAI's 81.3%. This is the biggest gap. Looking at the raw data, ARTEMIS debates often produced nuanced "it depends" style verdicts that the evaluator scored lower. Whether that's a flaw in ARTEMIS or in how we're measuring is unclear.

**Slower execution.** ARTEMIS averaged 87 seconds per debate. AutoGen did it in 31 seconds. The jury deliberation and multi-round evaluation add overhead. Some of this is inherent to the architecture, some is probably optimizable.

**The metrics don't capture our differentiators.** Safety monitoring, evidence tracking, causal reasoning chains - none of these showed up in the scores because the evaluator wasn't looking for them. We built features that this benchmark doesn't measure.

## Caveats Worth Mentioning

**LLM-as-judge is imperfect.** GPT-4o evaluating GPT-4o outputs has known biases. It tends to favor certain rhetorical styles. We used it because it's the standard approach, but take the absolute numbers with a grain of salt.

**The adapters aren't equivalent.** We wrote adapters to make each framework run debates, but they're not perfectly comparable. CrewAI is task-oriented, CAMEL is role-playing focused, AutoGen is conversation-based. We tried to make them as similar as possible, but there's inherent impedance mismatch.

**Small sample size.** 15 runs per framework is enough to see patterns but not enough for statistical significance on smaller effects. The broad strokes are real, the decimal points aren't.

## What We Learned

1. **Raw benchmark scores aren't everything.** ARTEMIS wasn't built to win generic debate benchmarks. It was built for structured, auditable, safe decision-making. We should measure what we actually care about.

2. **Decision accuracy needs work.** The gap is too big to ignore. We need to look at why ARTEMIS verdicts are scoring lower and whether it's a real problem or a measurement artifact.

3. **Speed vs depth is a real tradeoff.** The jury mechanism adds value but also adds latency. For some use cases that's fine, for others it's not. We might need a "fast mode" that skips the full deliberation.

4. **We need better benchmarks.** Future versions should measure:
   - Safety intervention rate (did monitors catch anything?)
   - Verdict explanation quality
   - Evidence citation accuracy
   - Consistency across rephrased topics

## What's Next

For v1.1, we're focusing on:
- Investigating the decision accuracy gap
- Adding benchmark dimensions for safety and evidence quality
- Optimizing the jury deliberation latency
- Better prompt engineering for the evaluator

The framework works. Now we need to make it better at the things it's supposed to be good at.

---

*Raw benchmark data available in `results/benchmark_20251229_032416.json`*
