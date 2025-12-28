# CrewAI Workflow Examples

This example demonstrates comprehensive CrewAI integration patterns using ARTEMIS debates within crew workflows.

## Basic Research-Debate-Decide Crew

```python
from crewai import Agent, Task, Crew, Process
from artemis.integrations.crewai import ArtemisCrewTool

# Create the debate tool
debate_tool = ArtemisCrewTool(
    model="gpt-4o",
    default_rounds=3,
)

# Define agents
researcher = Agent(
    role="Research Analyst",
    goal="Gather comprehensive information on the topic",
    backstory="""You are an expert researcher who excels at finding
    relevant data, statistics, and expert opinions on any topic.
    You provide balanced, factual information.""",
    tools=[],  # Add search tools as needed
    verbose=True,
)

debater = Agent(
    role="Debate Analyst",
    goal="Analyze topics through structured multi-perspective debate",
    backstory="""You are an expert at using the ARTEMIS debate framework
    to analyze complex topics. You run debates and synthesize insights
    from multiple perspectives.""",
    tools=[debate_tool.as_crewai_tool()],
    verbose=True,
)

decision_maker = Agent(
    role="Strategic Decision Maker",
    goal="Make clear recommendations based on analysis",
    backstory="""You are a senior executive who excels at synthesizing
    research and debate outcomes into actionable recommendations.
    You consider risks, opportunities, and stakeholder impact.""",
    tools=[],
    verbose=True,
)

# Define tasks
research_task = Task(
    description="""
    Research the following topic: {topic}

    Gather:
    - Key statistics and data points
    - Expert opinions from multiple perspectives
    - Recent developments and trends
    - Potential risks and opportunities

    Provide a comprehensive research brief.
    """,
    agent=researcher,
    expected_output="A detailed research brief with data and perspectives",
)

debate_task = Task(
    description="""
    Using the research findings, run a structured debate on: {topic}

    Use the debate tool to:
    1. Start a 3-round debate
    2. Analyze the verdict and key arguments
    3. Summarize the debate outcome

    Focus on identifying the strongest arguments on each side.
    """,
    agent=debater,
    expected_output="Debate verdict with key arguments from each side",
    context=[research_task],
)

decision_task = Task(
    description="""
    Based on the research and debate, provide a strategic recommendation.

    Include:
    - Clear recommendation (go/no-go, or specific approach)
    - Key supporting arguments
    - Risks and mitigation strategies
    - Implementation considerations
    - Confidence level and caveats
    """,
    agent=decision_maker,
    expected_output="Strategic recommendation with supporting analysis",
    context=[research_task, debate_task],
)

# Create and run crew
crew = Crew(
    agents=[researcher, debater, decision_maker],
    tasks=[research_task, debate_task, decision_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={
    "topic": "Should we migrate our data infrastructure to a lakehouse architecture?"
})

print(result)
```

## Multi-Debate Analysis Crew

```python
from crewai import Agent, Task, Crew, Process
from artemis.integrations.crewai import ArtemisCrewTool

debate_tool = ArtemisCrewTool(model="gpt-4o")

# Specialized debate agents for different aspects
technical_debater = Agent(
    role="Technical Analyst",
    goal="Analyze technical feasibility and architecture",
    backstory="""You focus on technical aspects: performance, scalability,
    reliability, and engineering complexity. You run debates specifically
    on technical trade-offs.""",
    tools=[debate_tool.as_crewai_tool()],
)

business_debater = Agent(
    role="Business Analyst",
    goal="Analyze business impact and ROI",
    backstory="""You focus on business aspects: costs, revenue impact,
    time-to-market, and competitive advantage. You run debates on
    business trade-offs.""",
    tools=[debate_tool.as_crewai_tool()],
)

risk_debater = Agent(
    role="Risk Analyst",
    goal="Analyze risks and mitigation strategies",
    backstory="""You focus on risk aspects: security, compliance,
    operational risks, and failure modes. You run debates on
    risk trade-offs.""",
    tools=[debate_tool.as_crewai_tool()],
)

synthesizer = Agent(
    role="Chief Strategy Officer",
    goal="Synthesize all analyses into coherent recommendation",
    backstory="""You are a senior executive who weighs technical,
    business, and risk factors to make balanced decisions.""",
    tools=[],
)

# Parallel debate tasks
tech_debate = Task(
    description="""
    Run a technical debate on: {topic}

    Focus on:
    - Technical architecture trade-offs
    - Performance and scalability
    - Implementation complexity
    - Technical debt implications
    """,
    agent=technical_debater,
    expected_output="Technical debate verdict and key findings",
)

business_debate = Task(
    description="""
    Run a business-focused debate on: {topic}

    Focus on:
    - ROI and cost analysis
    - Time-to-market impact
    - Competitive implications
    - Resource requirements
    """,
    agent=business_debater,
    expected_output="Business debate verdict and key findings",
)

risk_debate = Task(
    description="""
    Run a risk-focused debate on: {topic}

    Focus on:
    - Security implications
    - Compliance requirements
    - Operational risks
    - Mitigation strategies
    """,
    agent=risk_debater,
    expected_output="Risk debate verdict and key findings",
)

synthesis_task = Task(
    description="""
    Synthesize the technical, business, and risk analyses.

    Create a comprehensive recommendation that:
    - Weighs all three perspectives
    - Identifies conflicts and trade-offs
    - Provides a clear path forward
    - Includes contingency plans
    """,
    agent=synthesizer,
    expected_output="Comprehensive strategic recommendation",
    context=[tech_debate, business_debate, risk_debate],
)

crew = Crew(
    agents=[technical_debater, business_debater, risk_debater, synthesizer],
    tasks=[tech_debate, business_debate, risk_debate, synthesis_task],
    process=Process.sequential,  # Or hierarchical with manager
)

result = crew.kickoff(inputs={
    "topic": "Adopting Kubernetes for our legacy application infrastructure"
})
```

## Multi-Agent Debates with CrewAI

```python
from crewai import Agent, Task, Crew, Process
from artemis.integrations.crewai import ArtemisCrewTool

# Create tool that supports multi-agent debates
debate_tool = ArtemisCrewTool(
    model="gpt-4o",
    default_rounds=2,
)

debate_facilitator = Agent(
    role="Debate Facilitator",
    goal="Run multi-perspective debates using ARTEMIS",
    backstory="""You facilitate debates with multiple agents representing
    different viewpoints. You use the debate tool to get balanced analysis.""",
    tools=[debate_tool.as_crewai_tool()],
)

# Task using multi-agent debate
multi_agent_task = Task(
    description="""
    Run a multi-agent debate on: {topic}

    Use the debate tool with multiple agents:
    - agents: [
        {{"name": "architect", "role": "System Architect", "position": "focus on technical excellence"}},
        {{"name": "pragmatist", "role": "Pragmatic Engineer", "position": "focus on simplicity"}},
        {{"name": "security", "role": "Security Expert", "position": "focus on security first"}}
      ]
    - rounds: 2

    Summarize the verdict and key arguments from each perspective.
    """,
    agent=debate_facilitator,
    expected_output="Multi-perspective debate analysis",
)

crew = Crew(
    agents=[debate_facilitator],
    tasks=[multi_agent_task],
)

result = crew.kickoff(inputs={
    "topic": "How should we implement authentication for our new API?"
})
```

## Hierarchical Crew with Debate Manager

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from artemis.integrations.crewai import ArtemisCrewTool

debate_tool = ArtemisCrewTool(model="gpt-4o", default_rounds=2)

# Worker agents
fact_checker = Agent(
    role="Fact Checker",
    goal="Verify claims and gather evidence",
    backstory="Expert at finding and verifying factual information.",
    tools=[],  # search tools
)

perspective_analyst = Agent(
    role="Perspective Analyst",
    goal="Identify and articulate different viewpoints",
    backstory="Expert at understanding different stakeholder perspectives.",
    tools=[],
)

debate_runner = Agent(
    role="Debate Facilitator",
    goal="Run structured debates using ARTEMIS",
    backstory="Expert at facilitating productive debates.",
    tools=[debate_tool.as_crewai_tool()],
)

report_writer = Agent(
    role="Report Writer",
    goal="Create clear, actionable reports",
    backstory="Expert at synthesizing complex analysis into clear reports.",
    tools=[],
)

# Tasks
gather_facts = Task(
    description="Gather verified facts about: {topic}",
    agent=fact_checker,
    expected_output="Verified facts and data points",
)

identify_perspectives = Task(
    description="Identify key stakeholder perspectives on: {topic}",
    agent=perspective_analyst,
    expected_output="List of stakeholder perspectives and their concerns",
)

run_debate = Task(
    description="""
    Run a structured debate on {topic} incorporating:
    - The verified facts
    - The identified stakeholder perspectives

    Ensure the debate addresses all major viewpoints.
    """,
    agent=debate_runner,
    expected_output="Debate verdict with comprehensive analysis",
    context=[gather_facts, identify_perspectives],
)

write_report = Task(
    description="""
    Write an executive report on the debate findings.

    Include:
    - Executive summary
    - Key findings from each perspective
    - Debate verdict and confidence
    - Recommended actions
    - Appendix with detailed arguments
    """,
    agent=report_writer,
    expected_output="Executive report with recommendations",
    context=[run_debate],
)

# Hierarchical crew with manager
crew = Crew(
    agents=[fact_checker, perspective_analyst, debate_runner, report_writer],
    tasks=[gather_facts, identify_perspectives, run_debate, write_report],
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4o"),
)

result = crew.kickoff(inputs={
    "topic": "Implementing a four-day work week company-wide"
})
```

## Competitive Analysis Crew

```python
from crewai import Agent, Task, Crew, Process
from artemis.integrations.crewai import ArtemisCrewTool

debate_tool = ArtemisCrewTool(model="gpt-4o")

# Agents for competitive analysis
market_analyst = Agent(
    role="Market Analyst",
    goal="Analyze market dynamics and competitive landscape",
    backstory="Expert in market research and competitive intelligence.",
    tools=[],  # market research tools
)

war_gamer = Agent(
    role="War Gaming Facilitator",
    goal="Facilitate competitive war games through debate",
    backstory="""You run debates that simulate competitive scenarios
    to stress-test strategies.""",
    tools=[debate_tool.as_crewai_tool()],
)

strategist = Agent(
    role="Internal Strategist",
    goal="Develop and defend our strategic options",
    backstory="""You develop strategic options for our company and
    can argue their merits in competitive context.""",
    tools=[],
)

# Tasks
market_analysis = Task(
    description="""
    Analyze the competitive landscape for: {market}

    Identify:
    - Key competitors and their strategies
    - Market trends and dynamics
    - Competitive advantages and weaknesses
    - Emerging threats and opportunities
    """,
    agent=market_analyst,
    expected_output="Competitive landscape analysis",
)

strategy_debate = Task(
    description="""
    Run a war-gaming debate simulating competitive scenarios.

    Debate question: "What is the optimal strategy for {market}?"

    Consider:
    - Our current strategy vs alternatives
    - Competitor responses to our moves
    - Market evolution scenarios
    """,
    agent=war_gamer,
    expected_output="War game debate results with strategic insights",
    context=[market_analysis],
)

strategy_recommendation = Task(
    description="""
    Based on the war game results, develop a strategic recommendation.

    Include:
    - Recommended strategy
    - Competitive positioning
    - Key moves and timing
    - Response plans for competitor actions
    """,
    agent=strategist,
    expected_output="Strategic recommendation with competitive playbook",
    context=[market_analysis, strategy_debate],
)

crew = Crew(
    agents=[market_analyst, war_gamer, strategist],
    tasks=[market_analysis, strategy_debate, strategy_recommendation],
    process=Process.sequential,
)

result = crew.kickoff(inputs={
    "market": "enterprise AI development tools"
})
```

## Product Decision Crew

```python
from crewai import Agent, Task, Crew, Process
from artemis.integrations.crewai import ArtemisCrewTool

# Debate tool
debate_tool = ArtemisCrewTool(
    model="gpt-4o",
    default_rounds=2,
)

# Product team agents
product_manager = Agent(
    role="Product Manager",
    goal="Define product requirements and prioritize features",
    backstory="Expert at understanding customer needs and market fit.",
    tools=[],
)

engineering_lead = Agent(
    role="Engineering Lead",
    goal="Assess technical feasibility and effort",
    backstory="Expert at technical architecture and estimation.",
    tools=[],
)

design_lead = Agent(
    role="Design Lead",
    goal="Ensure great user experience",
    backstory="Expert at user-centered design and usability.",
    tools=[],
)

feature_debater = Agent(
    role="Feature Analyst",
    goal="Analyze feature trade-offs through debate",
    backstory="""You run debates to analyze feature decisions from
    multiple angles: user value, technical complexity, and business impact.""",
    tools=[debate_tool.as_crewai_tool()],
)

# Tasks for feature prioritization
gather_requirements = Task(
    description="""
    Gather requirements for: {feature}

    Document:
    - User stories and use cases
    - Success metrics
    - Dependencies and constraints
    """,
    agent=product_manager,
    expected_output="Feature requirements document",
)

technical_assessment = Task(
    description="""
    Assess technical feasibility of: {feature}

    Evaluate:
    - Architecture impact
    - Development effort
    - Technical risks
    - Dependencies
    """,
    agent=engineering_lead,
    expected_output="Technical feasibility assessment",
    context=[gather_requirements],
)

ux_assessment = Task(
    description="""
    Assess UX implications of: {feature}

    Evaluate:
    - User journey impact
    - Usability considerations
    - Design effort
    - Accessibility requirements
    """,
    agent=design_lead,
    expected_output="UX assessment",
    context=[gather_requirements],
)

feature_debate = Task(
    description="""
    Run a prioritization debate for: {feature}

    Debate question: "Should we prioritize this feature for the next quarter?"

    Consider inputs from:
    - Product requirements
    - Technical assessment
    - UX assessment

    Run a 2-round debate and provide recommendation.
    """,
    agent=feature_debater,
    expected_output="Feature prioritization recommendation with debate analysis",
    context=[gather_requirements, technical_assessment, ux_assessment],
)

crew = Crew(
    agents=[product_manager, engineering_lead, design_lead, feature_debater],
    tasks=[gather_requirements, technical_assessment, ux_assessment, feature_debate],
    process=Process.sequential,
)

result = crew.kickoff(inputs={
    "feature": "AI-powered search with natural language queries"
})
```

## Async Crew Execution

```python
import asyncio
from crewai import Agent, Task, Crew, Process
from artemis.integrations.crewai import ArtemisCrewTool

async def run_async_crew():
    debate_tool = ArtemisCrewTool(model="gpt-4o")

    analyst = Agent(
        role="Analyst",
        goal="Analyze topics through structured debate",
        tools=[debate_tool.as_crewai_tool()],
    )

    task = Task(
        description="Run a debate on: {topic}",
        agent=analyst,
        expected_output="Debate verdict and analysis",
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
    )

    # Run multiple analyses concurrently
    topics = [
        "Should we adopt GraphQL over REST?",
        "Should we move to a monorepo?",
        "Should we implement feature flags?",
    ]

    tasks = [
        crew.kickoff_async(inputs={"topic": topic})
        for topic in topics
    ]

    results = await asyncio.gather(*tasks)

    for topic, result in zip(topics, results):
        print(f"\n{topic}")
        print(f"Result: {result}")

asyncio.run(run_async_crew())
```

## Using DebateAnalyzer for Complex Decisions

```python
from artemis.integrations.crewai import DebateAnalyzer

# Use the high-level analyzer for multi-faceted decisions
analyzer = DebateAnalyzer(
    model="gpt-4o",
    rounds_per_debate=2,
)

# Single comprehensive analysis
result = analyzer.analyze_decision(
    decision="Should we migrate to cloud infrastructure?",
)

print(f"Verdict: {result['overall_verdict']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Recommendation: {result['recommendation']}")

# Multi-aspect analysis
result = analyzer.analyze_decision(
    decision="Should we adopt a new CRM platform?",
    aspects=["cost", "functionality", "integration", "user_adoption"],
)

print("\nMulti-Aspect Analysis:")
print(f"Decision: {result['decision']}")
print(f"Overall Verdict: {result['overall_verdict']}")
print(f"Verdict Distribution: {result['verdict_distribution']}")

for aspect, data in result["aspect_results"].items():
    print(f"\n{aspect.upper()}:")
    print(f"  Verdict: {data['verdict']}")
    print(f"  Confidence: {data['confidence']:.0%}")
```

## Next Steps

- See [Basic Debate](basic-debate.md) to understand ARTEMIS fundamentals
- Explore [Multi-Agent Debates](multi-agent.md) for complex scenarios
- Add [Safety Monitors](safety-monitors.md) to your crews
