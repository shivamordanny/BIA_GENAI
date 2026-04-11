# Session 8: Self-Reflection & Critique
## A Practical Guide to Making LLMs Evaluate and Improve Their Own Output

---

## 1. Quick Recap: From Session 7 to Session 8

In Session 7, you learned how to **structure** model reasoning. In Session 8, you learn how models can **improve** their own reasoning through critique and reflection.

| Aspect | Session 7: Conceptual Patterns | Session 8: Implementation |
|--------|-------------------------------|--------------------------|
| **Focus** | How to organize thinking paths | How to evaluate and refine output |
| **Patterns** | Chain-of-Thought (CoT), ReAct, Tree-of-Thought (ToT), Graph-of-Thought (GoT) | Basic Critique, Self-Refine, Reflexion |
| **Key Question** | "How should the model reason?" | "How can the model know if it's wrong?" |
| **Typical Use Case** | Complex multi-step problem solving | Quality improvement, error detection, iterative refinement |
| **Loop** | Single forward pass with structure | Generate → Critique → Revise → Repeat |

**Why this transition matters:** A structured prompt can guide reasoning, but LLMs don't inherently know when they produce incorrect or low-quality output. Session 8 teaches the model to critique itself—to become a judge of its own work.

---

## 2. Why Self-Reflection Matters

### The Core Problem

LLMs generate output based on patterns in training data. They have no built-in mechanism to:
- Detect their own mistakes
- Recognize incomplete reasoning
- Know when a response could be better

**Example:** An LLM generates a Python function that looks reasonable but has a subtle bug. Without external feedback, the model doesn't know it's wrong.

### The Solution: Self-Reflection

Self-reflection is the practice of asking a model to **evaluate and improve its own output**. It's analogous to how a human writer revises their own draft:

1. Write a first draft
2. Read it critically
3. Identify problems
4. Revise
5. Repeat until satisfied

### Three Levels of Self-Reflection

| Level | Pattern | Mechanism | When to Use |
|-------|---------|-----------|------------|
| **Level 1** | Basic Critique | Generate → Ask model to critique the output | Quick feedback, simple improvements |
| **Level 2** | Self-Refine | Generate → Critique → Refine → Repeat (N times) | Iterative quality improvement with stopping criteria |
| **Level 3** | Reflexion | Actor → Evaluator → Reflect → Memory → Retry | Learning over multiple attempts, tool-using agents |

### The Self-Reflection Loop

```
┌─────────────┐
│   Generate  │
│   Output    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Critique   │
│   Output    │
└──────┬──────┘
       │
       ▼
    Good?
    /   \
  Yes    No
  │       │
  │       ▼
  │    ┌─────────┐
  │    │ Revise  │
  │    │ Output  │
  │    └────┬────┘
  │         │
  │         └─────┐
  │               │
  └───────────────┘
        │
        ▼
    Output (Final)
```

**Why this matters:** Self-reflection can dramatically improve output quality. Studies show that even simple critique-and-refine loops reduce errors and improve coherence by 20-40% (Madaan et al., 2023).

---

## 3. Basic Self-Critique Pattern

The simplest self-reflection approach: generate output, then ask the model to critique it.

### Architecture

```python
Prompt (Task + Original Output)
    │
    ▼
LLM Critique Chain
    │
    ▼
Structured Critique Output
(issues, severity, suggestions)
```

### Implementation with LangChain

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define the critique schema
class CritiqueItem(BaseModel):
    issue: str = Field(description="A specific problem or weakness identified")
    severity: str = Field(
        description="How serious this issue is: 'critical', 'major', or 'minor'"
    )
    suggestion: str = Field(description="How to fix or improve this issue")

class Critique(BaseModel):
    overall_score: int = Field(
        description="Score from 1-10 for output quality",
        ge=1, le=10
    )
    issues: List[CritiqueItem] = Field(default_factory=list)
    strengths: List[str] = Field(
        description="What the output does well",
        default_factory=list
    )

# Set up the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Create the critique chain
critique_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a critical evaluator. Review the given output carefully.
Identify specific issues, rate their severity, and suggest improvements.
Be constructive but honest. Look for:
- Accuracy and correctness
- Clarity and organization
- Completeness
- Tone and appropriateness"""),
    ("user", """Review this output:

TASK: {task}

OUTPUT:
{output}

Provide a structured critique.""")
])

critique_chain = (
    critique_prompt
    | llm.with_structured_output(Critique)
)

# Example: Critique a code snippet
task = "Write a function to calculate factorial"
output = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

critique = critique_chain.invoke({
    "task": task,
    "output": output
})

print(f"Score: {critique.overall_score}/10")
for issue in critique.issues:
    print(f"[{issue.severity.upper()}] {issue.issue}")
    print(f"  → {issue.suggestion}")
for strength in critique.strengths:
    print(f"✓ {strength}")
```

### Example 2: Critique a Business Email

```python
task = "Write an email to request a deadline extension"
output = """
Subject: I need more time

Hi,

I need more time to finish the project. Can you give me an extension?
I have other work to do.

Thanks
"""

critique = critique_chain.invoke({
    "task": task,
    "output": output
})

# Output might look like:
# Score: 3/10
# [CRITICAL] Vague subject line doesn't explain urgency
# [MAJOR] No specific deadline requested
# [MAJOR] Doesn't explain reason professionally
# etc.
```

### When to Use Basic Critique

- **Pros:** Simple, fast, minimal API calls
- **Cons:** No iterative improvement, single evaluation pass
- **Use case:** Quick quality checks, feedback generation

---

## 4. Iterative Refinement (Self-Refine)

Self-Refine is a loop where the model generates output, receives feedback, refines, and repeats until quality is good or stopping criteria are met.

**Based on:** Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback" (ICLR 2024)

### The Algorithm

```
For iteration i = 1 to max_iterations:
  1. output_i = LLM(prompt, feedback_history)
  2. feedback_i = LLM_evaluate(task, output_i)
  3. If feedback_i.quality_score >= threshold:
       return output_i
  4. If no_new_issues(feedback_i, feedback_history):
       return output_i
  5. Append feedback_i to feedback_history
```

### Full Implementation with LangChain

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Schemas
class Feedback(BaseModel):
    quality_score: int = Field(
        description="Quality rating 1-10", ge=1, le=10
    )
    issues: List[str] = Field(
        description="List of issues to fix"
    )
    actionable_items: List[str] = Field(
        description="Specific things to do in the next iteration"
    )

class RefinedOutput(BaseModel):
    content: str = Field(description="The refined output")
    iteration: int = Field(description="Which iteration this is")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Chain 1: Generate/Refine
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert writer and problem solver.
Create high-quality output based on the task and any feedback provided.
Incorporate all actionable feedback from previous iterations.
Aim for clarity, accuracy, and completeness."""),
    ("user", """TASK: {task}

{feedback_context}

Create or refine the output:""")
])

refine_chain = (
    refine_prompt
    | llm.with_structured_output(RefinedOutput)
)

# Chain 2: Evaluate
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a quality evaluator. Assess the output and provide
constructive feedback for the next iteration."""),
    ("user", """TASK: {task}

OUTPUT:
{output}

Provide feedback to improve this output.""")
])

eval_chain = (
    eval_prompt
    | llm.with_structured_output(Feedback)
)

# Main Self-Refine Loop
def self_refine(
    task: str,
    max_iterations: int = 3,
    quality_threshold: int = 8
) -> tuple[str, int, list]:
    """
    Iteratively refine output until quality threshold or max iterations.
    Returns: (final_output, iterations_used, feedback_history)
    """

    feedback_history = []
    current_output = None

    for iteration in range(1, max_iterations + 1):
        # Build context from feedback
        feedback_context = ""
        if feedback_history:
            feedback_context = "FEEDBACK FROM PREVIOUS ITERATION(S):\n"
            for i, fb in enumerate(feedback_history[-1:]):  # Use only latest
                feedback_context += f"Issues: {', '.join(fb.issues)}\n"
                feedback_context += f"Next steps: {', '.join(fb.actionable_items)}\n"

        # Generate/refine
        refined = refine_chain.invoke({
            "task": task,
            "feedback_context": feedback_context
        })
        current_output = refined.content

        print(f"\n[Iteration {iteration}]")
        print(f"Output length: {len(current_output)} chars")

        # Evaluate
        feedback = eval_chain.invoke({
            "task": task,
            "output": current_output
        })
        feedback_history.append(feedback)

        print(f"Quality score: {feedback.quality_score}/10")
        print(f"Issues: {feedback.issues}")

        # Check stopping criteria
        if feedback.quality_score >= quality_threshold:
            print(f"\n✓ Quality threshold reached!")
            break

        if iteration > 1:
            # Check if we're making progress
            prev_issues = set(feedback_history[-2].issues)
            curr_issues = set(feedback.issues)
            if prev_issues == curr_issues:
                print(f"\n✓ No new issues found. Stopping.")
                break

    return current_output, len(feedback_history), feedback_history

# Example: Improve a technical explanation
task = """Explain how neural networks learn,
in terms a non-technical person can understand."""

final_output, iterations, history = self_refine(
    task=task,
    max_iterations=3,
    quality_threshold=8
)

print(f"\n\nFINAL OUTPUT (after {iterations} iterations):")
print(final_output)
```

### Stopping Criteria in Detail

**1. Quality Threshold**
```python
if feedback.quality_score >= threshold:
    return output  # Good enough!
```

**2. No Progress**
```python
if prev_feedback.issues == curr_feedback.issues:
    return output  # Same issues, stop iterating
```

**3. Max Iterations**
```python
if iteration >= max_iterations:
    return output  # Hit the limit
```

### When to Use Self-Refine

- **Pros:** Improves output over iterations, simple loop, measurable progress
- **Cons:** Multiple API calls (can be expensive), slower than single pass
- **Use case:** Content generation (essays, explanations, code), quality-critical tasks

---

## 5. Reflexion Algorithm (Deep Dive)

Reflexion is a more sophisticated approach where an agent learns from failures by storing reflections in memory, then using them to improve future attempts.

**Based on:** Shinn, Cassano, et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)

### Architecture

```
┌──────────────┐
│    Actor     │
│  (Generate   │
│   Response)  │◄────────────────┐
└──────┬───────┘                │
       │                        │
       ▼                        │
┌──────────────────┐            │
│   Evaluator      │            │
│  (Score, Pass?)  │            │
└──────┬───────────┘            │
       │                        │
       ├─ Success? ──────→ DONE │
       │                        │
       └─ Failure ──→ ┌─────────┴─────────┐
                      │ Self-Reflection   │
                      │  (Analyze error,  │
                      │  Learn insight)   │
                      └──────┬────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Memory Buffer│
                      │ (Store text) │
                      └──────┬───────┘
                             │
                             └─────────→ (Append to Actor prompt)
```

### Key Insight: Verbal Reinforcement Learning

Unlike traditional RL which uses numerical gradients, Reflexion uses **text-based learning signals**:

- **Failure:** "The function doesn't handle negative inputs"
- **Insight:** "Need to add input validation before processing"
- **Memory:** Store this as text and feed it to future attempts

This allows LLMs to learn conceptually, not just numerically.

### Full Implementation

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Schemas
class ActorOutput(BaseModel):
    response: str = Field(description="The generated response")

class EvaluatorOutput(BaseModel):
    passed: bool = Field(description="Did the response succeed?")
    score: float = Field(description="Quality score 0-1")
    reasoning: str = Field(
        description="Why it passed or failed"
    )

class ReflectionOutput(BaseModel):
    failure_analysis: str = Field(
        description="What went wrong and why"
    )
    actionable_insight: str = Field(
        description="A specific lesson to remember"
    )

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Chain 1: Actor (Generate response)
actor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant solving problems.
Generate a clear, correct response to the task.

{memory_context}

Remember past failures and apply learnings."""),
    ("user", "{task}")
])

actor_chain = (
    actor_prompt
    | llm.with_structured_output(ActorOutput)
)

# Chain 2: Evaluator (Judge response)
evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluator. Judge if the response
correctly solves the task. Be strict."""),
    ("user", """TASK: {task}

RESPONSE:
{response}

Evaluate this response.""")
])

evaluator_chain = (
    evaluator_prompt
    | llm.with_structured_output(EvaluatorOutput)
)

# Chain 3: Self-Reflection (Learn from failure)
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a reflective analyst. When a response fails,
analyze the failure and extract a concrete lesson for next time."""),
    ("user", """TASK: {task}

FAILED RESPONSE:
{response}

FAILURE REASON:
{failure_reason}

Analyze this failure and provide a lesson.""")
])

reflection_chain = (
    reflection_prompt
    | llm.with_structured_output(ReflectionOutput)
)

# Main Reflexion Agent
class ReflexionAgent:
    def __init__(self, task: str, max_trials: int = 3):
        self.task = task
        self.max_trials = max_trials
        self.memory = []  # Store reflection strings
        self.trial_history = []

    def _format_memory_context(self) -> str:
        """Format memory for inclusion in actor prompt"""
        if not self.memory:
            return ""
        return "LEARNED LESSONS FROM PAST FAILURES:\n" + "\n".join(
            [f"- {m}" for m in self.memory]
        )

    def run(self) -> dict:
        """Execute the Reflexion loop"""

        for trial in range(1, self.max_trials + 1):
            print(f"\n{'='*60}")
            print(f"Trial {trial}/{self.max_trials}")
            print('='*60)

            # Step 1: Generate response (with learned memory)
            memory_context = self._format_memory_context()
            response = actor_chain.invoke({
                "task": self.task,
                "memory_context": memory_context
            })
            generated = response.response

            print(f"Generated response:\n{generated[:200]}...")

            # Step 2: Evaluate response
            evaluation = evaluator_chain.invoke({
                "task": self.task,
                "response": generated
            })

            print(f"Score: {evaluation.score:.2f}, Passed: {evaluation.passed}")
            print(f"Reasoning: {evaluation.reasoning}")

            self.trial_history.append({
                "trial": trial,
                "response": generated,
                "passed": evaluation.passed,
                "score": evaluation.score,
                "reasoning": evaluation.reasoning
            })

            # Check success
            if evaluation.passed:
                print(f"\n✓ SUCCESS after {trial} trial(s)!")
                return {
                    "success": True,
                    "final_response": generated,
                    "trials_used": trial,
                    "history": self.trial_history,
                    "memory_used": self.memory
                }

            # Step 3: Reflect on failure and learn
            if trial < self.max_trials:
                reflection = reflection_chain.invoke({
                    "task": self.task,
                    "response": generated,
                    "failure_reason": evaluation.reasoning
                })

                print(f"Failure analysis: {reflection.failure_analysis}")
                print(f"Learned: {reflection.actionable_insight}")

                # Store in memory for next trial
                self.memory.append(reflection.actionable_insight)

        # Failed after all trials
        print(f"\n✗ Failed to solve after {self.max_trials} trials")
        return {
            "success": False,
            "final_response": self.trial_history[-1]["response"],
            "trials_used": self.max_trials,
            "history": self.trial_history,
            "memory_used": self.memory
        }

# Example: Solve a coding challenge with Reflexion
task = """Write a function that checks if a string is a valid email.
Requirements:
1. Must contain @ symbol
2. Must have text before @
3. Must have domain after @
4. Domain must contain a dot
5. Must not contain spaces"""

agent = ReflexionAgent(task=task, max_trials=3)
result = agent.run()

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(f"Success: {result['success']}")
print(f"Trials: {result['trials_used']}")
print(f"Lessons learned: {result['memory_used']}")
```

### Reflexion Memory in Action

Trial 1 might fail because the function doesn't validate dots in domain.
- **Reflection:** "Must explicitly check that domain has a dot"
- **Memory:** Store this insight
- **Trial 2:** Actor now sees this memory and generates a better function

This simulates learning through experience—exactly what makes Reflexion powerful.

### When to Use Reflexion

- **Pros:** Learns over multiple attempts, suitable for complex tasks, agent-based systems
- **Cons:** Requires good evaluator, many API calls, slower
- **Use case:** Coding challenges, multi-step reasoning, tool-using agents that need to learn

---

## 6. LLM-as-Judge Evaluator

The quality of self-reflection depends on the evaluator. This section covers patterns for building strong evaluators.

### Pattern 1: Single-Point Scoring with Rubric

Rate output on a 1-5 scale with explicit criteria.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class RubricScore(BaseModel):
    clarity: int = Field(
        description="1-5: Is the output clear and easy to understand?",
        ge=1, le=5
    )
    accuracy: int = Field(
        description="1-5: Is the content factually correct?",
        ge=1, le=5
    )
    completeness: int = Field(
        description="1-5: Does it address all aspects of the task?",
        ge=1, le=5
    )
    overall: int = Field(
        description="1-5: Overall quality",
        ge=1, le=5
    )
    reasoning: str = Field(
        description="Explain your scores"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

rubric_prompt = ChatPromptTemplate.from_messages([
    ("system", """Score the output using this rubric:
- Clarity: Is it easy to understand?
- Accuracy: Are facts correct?
- Completeness: Does it fully address the task?
Be objective and consistent."""),
    ("user", """TASK: {task}

OUTPUT:
{output}

Score using the rubric.""")
])

rubric_chain = (
    rubric_prompt
    | llm.with_structured_output(RubricScore)
)

# Usage
score = rubric_chain.invoke({
    "task": "Explain photosynthesis",
    "output": "Plants use light to make food..."
})

print(f"Clarity: {score.clarity}/5")
print(f"Accuracy: {score.accuracy}/5")
print(f"Overall: {score.overall}/5")
print(f"Notes: {score.reasoning}")
```

### Pattern 2: Reference-Based Evaluation

Compare output to a gold standard answer.

```python
class ReferenceComparison(BaseModel):
    matches_gold: bool = Field(
        description="Does the output align with the gold standard?"
    )
    missing_points: list[str] = Field(
        description="Key points in gold standard but missing from output"
    )
    extra_points: list[str] = Field(
        description="Points in output not in gold standard (may be good)"
    )
    overall_score: float = Field(
        description="Similarity score 0-1",
        ge=0, le=1
    )

reference_prompt = ChatPromptTemplate.from_messages([
    ("system", """Compare the output to the gold standard answer.
Identify what's missing and what's extra.
Rate overall similarity."""),
    ("user", """TASK: {task}

GOLD STANDARD:
{gold_standard}

ACTUAL OUTPUT:
{output}

Compare them.""")
])

reference_chain = (
    reference_prompt
    | llm.with_structured_output(ReferenceComparison)
)

# Usage
comparison = reference_chain.invoke({
    "task": "What are the three branches of government?",
    "gold_standard": "Executive (President), Legislative (Congress), Judicial (Courts)",
    "output": "The Executive and Legislative branches..."
})

print(f"Match: {comparison.matches_gold}")
print(f"Missing: {comparison.missing_points}")
print(f"Score: {comparison.overall_score:.2f}")
```

### Pattern 3: Pairwise Comparison

Given two responses, determine which is better.

```python
class PairwiseComparison(BaseModel):
    reasoning: str = Field(
        description="Chain-of-thought reasoning before verdict"
    )
    winner: str = Field(
        description="Which response is better? 'A', 'B', or 'TIE'"
    )
    confidence: float = Field(
        description="How confident? 0-1",
        ge=0, le=1
    )

pairwise_prompt = ChatPromptTemplate.from_messages([
    ("system", """Compare two responses to the same task.
Think step-by-step about their strengths and weaknesses.
Then declare a winner."""),
    ("user", """TASK: {task}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which is better and why?""")
])

pairwise_chain = (
    pairwise_prompt
    | llm.with_structured_output(PairwiseComparison)
)

# Usage
comparison = pairwise_chain.invoke({
    "task": "Write a haiku about spring",
    "response_a": "Spring is here now / Green leaves on trees / I like it",
    "response_b": "Cherry blossoms bloom / Gentle breeze through morning air / New season awakens"
})

print(f"Reasoning:\n{comparison.reasoning}")
print(f"Winner: Response {comparison.winner}")
print(f"Confidence: {comparison.confidence:.0%}")
```

### Avoiding Evaluator Biases

LLMs as judges can exhibit systematic biases. Be aware:

| Bias | Description | Mitigation |
|------|-------------|-----------|
| **Position Bias** | Prefers first or last option | Randomize order, average multiple runs |
| **Verbosity Bias** | Prefers longer responses | Use length-normalized metrics |
| **Self-Enhancement Bias** | Prefers responses similar to its own style | Use diverse reference standards |
| **Anchoring Bias** | Early information influences later judgments | Use blind evaluation (hide author/prior scores) |

```python
# Mitigation: Run evaluation twice with swapped positions
eval_ab = pairwise_chain.invoke({
    "task": task,
    "response_a": response_a,
    "response_b": response_b
})

eval_ba = pairwise_chain.invoke({
    "task": task,
    "response_a": response_b,
    "response_b": response_a
})

# Aggregate results
if eval_ab.winner == eval_ba.winner:
    print(f"Robust winner: {eval_ab.winner}")
else:
    print(f"Position bias detected. Consider TIE")
```

---

## 7. Composing Patterns

Self-reflection patterns can be combined for greater power.

### Decision Tree: Which Pattern to Use?

```
Is this a single output that needs feedback?
├─ YES, simple task → Use BASIC CRITIQUE
└─ NO

Should the model improve iteratively?
├─ YES, with quality threshold → Use SELF-REFINE
└─ NO

Does the model need to learn from failures?
├─ YES, and retry → Use REFLEXION
└─ NO
```

### Combination 1: Self-Refine + LLM-as-Judge

Use Self-Refine but replace the simple evaluator with a sophisticated one.

```python
# Instead of the simple eval_chain in Section 4,
# use a pairwise or rubric-based evaluator

eval_chain = (
    rubric_prompt
    | llm.with_structured_output(RubricScore)
)

# In the self_refine function:
feedback = eval_chain.invoke({
    "task": task,
    "output": current_output
})

# Now quality threshold check uses rubric
if feedback.overall >= 4:  # 4+ out of 5
    break
```

### Combination 2: Reflexion + ReAct (from Session 7)

Build an agent that uses ReAct for tool-based reasoning AND learns from failures via Reflexion.

```python
# In the actor_chain, use ReAct format:

actor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an agent that solves problems using tools.
Use the ReAct format:
Thought: What should I do next?
Action: Use a tool
Observation: What did I learn?
... repeat ...
Final Answer: The solution

{memory_context}"""),
    ("user", "{task}")
])

# The evaluator checks if the final answer is correct
# On failure, Reflexion extracts lessons about which tool to use
# On retry, actor has memory: "Tool X doesn't work, try Tool Y"
```

### Combination 3: Chained Evaluators

Use multiple evaluators and combine scores.

```python
# Evaluator 1: Quick rubric check
quick_eval = rubric_chain.invoke({"task": task, "output": output})

# Evaluator 2: Reference-based check
ref_eval = reference_chain.invoke({
    "task": task,
    "gold_standard": gold_standard,
    "output": output
})

# Combine
combined_score = (quick_eval.overall + ref_eval.overall_score * 5) / 2

if combined_score >= threshold:
    return output
```

---

## 8. Hands-On Exercises

### Exercise 1: Build a Self-Improving Essay Writer

**Objective:** Create a system that generates an essay, critiques it, and refines it over multiple iterations.

**Starter Code:**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# TODO 1: Define a Feedback schema with:
#   - clarity_score (int, 1-10)
#   - organization_score (int, 1-10)
#   - specific_issues (List[str])
#   - improvement_suggestions (List[str])
class Feedback(BaseModel):
    clarity_score: int = Field(ge=1, le=10)
    organization_score: int = Field(ge=1, le=10)
    specific_issues: List[str]
    improvement_suggestions: List[str]

# TODO 2: Create essay_generation_prompt with ChatPromptTemplate
#   System: "You are an expert essay writer"
#   User: "Write a {word_count} word essay on: {topic}\n{feedback_context}"
essay_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "TODO: Add system message"),
    ("user", "TODO: Add user message template")
])

# TODO 3: Create essay_evaluation_prompt with ChatPromptTemplate
#   System: "You are an essay evaluator"
#   User: "Evaluate this essay on '{topic}':\n{essay}\nProvide structured feedback"
essay_evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", "TODO: Add system message"),
    ("user", "TODO: Add user message template")
])

# TODO 4: Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="TODO: Use gemini-2.5-flash",
    temperature=0.7
)

# TODO 5: Create chains
generation_chain = (
    essay_generation_prompt
    | llm.with_structured_output(TODO: Define output schema)
)

eval_chain = (
    essay_evaluation_prompt
    | llm.with_structured_output(Feedback)
)

# TODO 6: Implement essay_improver function
#   Parameters: topic (str), word_count (int), max_iterations (int)
#   Loop:
#     1. Generate essay (include feedback_context if iteration > 1)
#     2. Evaluate essay
#     3. Print scores and issues
#     4. Stop if both scores >= 8
#     5. Store feedback for next iteration
#   Return: (final_essay, num_iterations, feedback_history)
def essay_improver(topic: str, word_count: int, max_iterations: int = 3):
    feedback_history = []

    for iteration in range(1, max_iterations + 1):
        # TODO: Generate essay

        # TODO: Evaluate essay

        # TODO: Check stopping criteria

        pass

    return None, None, feedback_history

# TEST YOUR CODE
if __name__ == "__main__":
    essay, iterations, history = essay_improver(
        topic="The Future of Remote Work",
        word_count=300,
        max_iterations=3
    )

    print(f"\nFinal essay ({iterations} iterations):")
    print(essay)
```

**Stretch Goals:**
1. Add a maximum essay length check to prevent bloat
2. Track which specific issues were resolved across iterations
3. Create a "summary of changes" that shows before/after

---

### Exercise 2: Build an LLM-as-Judge for Code Quality

**Objective:** Create a pairwise code evaluator that compares two solutions and provides detailed reasoning.

**Starter Code:**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# TODO 1: Define CodeQualityAnalysis schema with:
#   - correctness_assessment (str): Does it solve the problem?
#   - efficiency_assessment (str): Time and space complexity
#   - readability_assessment (str): Code clarity and style
#   - robustness_assessment (str): Edge cases, error handling
#   - reasoning (str): Chain-of-thought before verdict
#   - winner (str): 'A' or 'B'
class CodeQualityAnalysis(BaseModel):
    correctness_assessment: str
    efficiency_assessment: str
    readability_assessment: str
    robustness_assessment: str
    reasoning: str
    winner: str = Field(description="'A' or 'B'")

# TODO 2: Create evaluation prompt with ChatPromptTemplate
#   System: "You are an expert code reviewer"
#   User: Include problem description, code A, code B, and evaluation request
code_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "TODO: Add system message"),
    ("user", "TODO: Add user message template with problem, code_a, code_b")
])

# TODO 3: Initialize LLM and create chain
llm = ChatGoogleGenerativeAI(
    model="TODO: gemini-2.5-flash",
    temperature=0.5  # Lower temp for consistency
)

evaluation_chain = (
    code_eval_prompt
    | llm.with_structured_output(CodeQualityAnalysis)
)

# TODO 4: Implement compare_solutions function
#   Parameters: problem (str), code_a (str), code_b (str)
#   1. Call evaluation_chain
#   2. Print all assessments with clear formatting
#   3. Show the winner and reasoning
#   Return: CodeQualityAnalysis object
def compare_solutions(problem: str, code_a: str, code_b: str):
    analysis = evaluation_chain.invoke({
        "TODO": "Complete the invocation"
    })

    # TODO: Pretty-print the analysis

    return analysis

# TODO 5: (Stretch) Implement avoid_position_bias
#   Run evaluation twice: (A vs B) and (B vs A)
#   Compare results
#   If winners differ, report both evaluations
def avoid_position_bias(problem: str, code_a: str, code_b: str):
    eval_ab = compare_solutions(problem, code_a, code_b)
    eval_ba = compare_solutions(problem, code_b, code_a)

    # TODO: Compare eval_ab.winner with eval_ba.winner
    # TODO: Print warning if different

    pass

# TEST YOUR CODE
if __name__ == "__main__":
    problem = """Write a function to find the second largest element in a list.
    Requirements:
    - Handle edge cases (less than 2 elements)
    - O(n) time complexity preferred
    - No built-in sort()"""

    code_a = """
def second_largest(lst):
    return sorted(lst, reverse=True)[1]
    """

    code_b = """
def second_largest(lst):
    if len(lst) < 2:
        return None
    max1 = max2 = float('-inf')
    for num in lst:
        if num > max1:
            max2 = max1
            max1 = num
        elif num > max2:
            max2 = num
    return max2 if max2 != float('-inf') else None
    """

    analysis = compare_solutions(problem, code_a, code_b)
    print(f"\nWinner: Code {analysis.winner}")
```

**Stretch Goals:**
1. Implement the `avoid_position_bias()` function
2. Run evaluation 3+ times and aggregate winners
3. Add a confidence score based on agreement across runs

---

## 9. Quick Reference

### Pattern Comparison Table

| Pattern | Mechanism | API Calls | Speed | Quality Gain | Complexity |
|---------|-----------|-----------|-------|--------------|-----------|
| **Basic Critique** | Generate → Critique | 2 | Fast | +20% | Low |
| **Self-Refine** | Generate → Evaluate → Refine (loop) | 3N | Slow | +30-40% | Medium |
| **Reflexion** | Actor → Eval → Reflect → Retry | 4N | Very Slow | +40-50% | High |

### Key LangChain Imports

```python
# Core
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# For structured output
from pydantic import BaseModel, Field
from typing import List

# Chains (LCEL)
# Use pipe operator: prompt | llm | parser
```

### LangChain Pattern: Structured Output

```python
# 1. Define schema
class MyOutput(BaseModel):
    field1: str = Field(description="...")
    field2: int = Field(ge=0, le=100)

# 2. Create chain with .with_structured_output()
chain = (
    prompt
    | llm.with_structured_output(MyOutput)
)

# 3. Invoke
result: MyOutput = chain.invoke({"var": "value"})
```

### Cost Considerations

Estimate API cost for different patterns:

**Self-Refine (3 iterations, max 1000 tokens per call):**
- 3 generations × 1000 tokens = 3,000 tokens
- 3 evaluations × 500 tokens = 1,500 tokens
- Total: 4,500 input tokens

**Reflexion (3 trials, each with 2-3 calls):**
- 9 calls × 750 avg tokens = 6,750 tokens
- More expensive but better for complex tasks

**Use Case → Pattern Selection Table**

| Use Case | Recommended Pattern | Reason |
|----------|-------------------|--------|
| Quick quality check | Basic Critique | Fast, minimal cost |
| Content generation | Self-Refine | Good quality/cost tradeoff |
| Problem solving | Reflexion | Learning improves success |
| Code review | LLM-as-Judge | Specialized evaluation |
| Real-time applications | Basic Critique | Speed critical |
| Batch/offline work | Reflexion | Cost less important |

---

## 10. Summary & Next Steps

### What You've Learned

1. **Basic Critique** - Generate then critique in one pass
2. **Self-Refine** - Iterative improvement with feedback loops
3. **Reflexion** - Learning from failure through memory
4. **LLM-as-Judge** - Building reliable evaluators
5. **Composition** - Combining patterns for greater power

### When to Use Self-Reflection

- ✓ Quality is critical (essays, code, technical writing)
- ✓ Single-shot generation isn't good enough
- ✓ You can afford multiple API calls
- ✗ Real-time systems (too slow)
- ✗ Simple tasks with clear answers
- ✗ When cost is paramount

### Next Steps

1. **Experiment:** Pick one pattern and implement it with your own task
2. **Measure:** Track quality improvements (before/after)
3. **Optimize:** Tune prompts, thresholds, iteration counts
4. **Combine:** Try mixing patterns from this session with Session 7 patterns

### Advanced Topics (Beyond This Session)

- **Constitutional AI:** Refining with explicit principles instead of evaluators
- **Multi-Agent Reflection:** Teams of agents critiquing each other
- **Human-in-the-Loop:** Combining LLM critique with human feedback
- **Reward Models:** Training a separate model to evaluate quality

---

## Appendix: Full Working Examples

### Example 1: Complete Self-Refine Pipeline

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

class Feedback(BaseModel):
    score: int = Field(ge=1, le=10)
    issues: List[str]
    suggestions: List[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define chains
write_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a clear, well-structured response."),
    ("user", "{task}\n\n{feedback_context}")
])

evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Evaluate the quality of this response."),
    ("user", "Task: {task}\n\nResponse: {response}")
])

write_chain = write_prompt | llm
eval_chain = evaluate_prompt | llm.with_structured_output(Feedback)

# Refine loop
def refine_response(task: str, max_iters: int = 3):
    feedback_list = []
    current_response = None

    for i in range(max_iters):
        feedback_ctx = ""
        if feedback_list:
            feedback_ctx = "FEEDBACK FROM PREVIOUS ITERATION:\n"
            feedback_ctx += f"Issues: {', '.join(feedback_list[-1].issues)}\n"
            feedback_ctx += f"Fix these: {', '.join(feedback_list[-1].suggestions)}"

        # Generate
        current_response = write_chain.invoke({
            "task": task,
            "feedback_context": feedback_ctx
        })

        # Evaluate
        feedback = eval_chain.invoke({
            "task": task,
            "response": current_response
        })
        feedback_list.append(feedback)

        print(f"[Iteration {i+1}] Score: {feedback.score}/10")

        if feedback.score >= 8:
            break

    return current_response, feedback_list

# Test
task = "Explain machine learning to a 10-year-old"
response, history = refine_response(task)
print(f"\nFinal response:\n{response}")
```

### Example 2: Reflexion Agent for Math Problems

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Solution(BaseModel):
    reasoning: str
    answer: str

class Evaluation(BaseModel):
    is_correct: bool
    feedback: str

class Lesson(BaseModel):
    insight: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

solve_prompt = ChatPromptTemplate.from_messages([
    ("system", "Solve the math problem step by step. {memory}"),
    ("user", "{problem}")
])

check_prompt = ChatPromptTemplate.from_messages([
    ("system", "Check if this solution is correct."),
    ("user", "Problem: {problem}\n\nSolution: {solution}")
])

reflect_prompt = ChatPromptTemplate.from_messages([
    ("system", "Learn from this mistake."),
    ("user", "Problem: {problem}\n\nError: {error}\n\nWhat should be done next time?")
])

solve_chain = solve_prompt | llm.with_structured_output(Solution)
check_chain = check_prompt | llm.with_structured_output(Evaluation)
reflect_chain = reflect_prompt | llm.with_structured_output(Lesson)

# Reflexion loop
def solve_with_reflexion(problem: str, max_attempts: int = 3):
    memory = []

    for attempt in range(max_attempts):
        memory_text = "\n".join([f"Lesson: {m}" for m in memory])

        solution = solve_chain.invoke({
            "problem": problem,
            "memory": memory_text
        })

        evaluation = check_chain.invoke({
            "problem": problem,
            "solution": solution.answer
        })

        if evaluation.is_correct:
            return solution.answer, attempt + 1, memory

        if attempt < max_attempts - 1:
            lesson = reflect_chain.invoke({
                "problem": problem,
                "error": evaluation.feedback
            })
            memory.append(lesson.insight)

    return solution.answer, max_attempts, memory

# Test
problem = "A train travels 100 miles in 2 hours. How far does it travel in 5 hours?"
answer, attempts, lessons = solve_with_reflexion(problem)
print(f"Answer: {answer}")
print(f"Attempts: {attempts}")
print(f"Lessons learned: {lessons}")
```

---

## Credits & Resources

- **Self-Refine**: Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback" (ICLR 2024)
- **Reflexion**: Shinn, Cassano, et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)
- **LangChain Documentation**: https://python.langchain.com/
- **Google Gemini API**: https://ai.google.dev/

---

**© BIA® School of Technology & AI — Generative AI & Agentic AI Development Program**

*Last Updated: 2026-04-04*
