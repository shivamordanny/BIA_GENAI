# Structured Prompting: From Reasoning Chains to Autonomous Agents
## Implementing Chain-of-Thought, ReAct, Tree-of-Thought & Graph-of-Thought

> **BIA® School of Technology & AI — Generative AI & Agentic AI Development**
> Session 7 Reference Material

---

## Table of Contents

1. [Recap: Where Session 6 Left Off](#1-recap-where-session-6-left-off)
2. [Chain-of-Thought in Practice](#2-chain-of-thought-in-practice)
3. [ReAct: Building Your First Agent Loop](#3-react-building-your-first-agent-loop)
4. [Tree-of-Thought: Multi-Path Reasoning](#4-tree-of-thought-multi-path-reasoning)
5. [Graph-of-Thought: Non-Linear Problem Solving](#5-graph-of-thought-non-linear-problem-solving)
6. [Choosing the Right Reasoning Pattern](#6-choosing-the-right-reasoning-pattern)
7. [Composing Patterns: Real-World Combinations](#7-composing-patterns-real-world-combinations)
8. [Hands-On Exercises](#8-hands-on-exercises)
9. [Quick Reference](#9-quick-reference)

---

## 1. Recap: Where Session 6 Left Off

In Session 6 we covered the *what* — what Chain-of-Thought, ReAct, and Tree-of-Thought are conceptually. Today we focus on the *how* — implementing these patterns in real Python code against the OpenAI API, understanding when each pattern shines, and composing them into agent workflows.

### What You Should Already Know

| Concept | Session 6 Coverage | Session 7 Goes Deeper |
|---------|-------------------|----------------------|
| Chain-of-Thought (CoT) | "Add 'think step by step' to your prompt" | Implementing CoT parsers, structured CoT, auto-CoT |
| ReAct | Thought → Action → Observation loop | Full agent loop with tool execution and error recovery |
| Tree-of-Thought | Explore multiple reasoning paths | Implementing breadth-first and depth-first ToT |
| Graph-of-Thought | Brief mention | Full implementation — merging, looping, dependency tracking |

### Setup for Today

All code today uses the OpenAI Python SDK with `gpt-4o-mini`. Make sure you have this ready:

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def call_llm(messages, temperature=0.0, max_tokens=1000, model="gpt-4o-mini"):
    """Helper function we'll use throughout this session."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
```

---

## 2. Chain-of-Thought in Practice

### 2.1 Beyond "Think Step by Step"

In Session 6, you saw that adding "think step by step" dramatically improves reasoning. But in production, you need more control. You need to **parse** the reasoning, **validate** each step, and sometimes **inject** information between steps.

**The Three Levels of CoT:**

```
Level 1: Zero-shot CoT    → "Think step by step" (Session 6)
Level 2: Structured CoT   → Force specific reasoning steps with a schema
Level 3: Programmatic CoT → Parse and validate each reasoning step in code
```

### 2.2 Structured Chain-of-Thought

Instead of letting the model free-form its reasoning, define the exact structure you want.

```python
STRUCTURED_COT_SYSTEM = """You are an analytical problem solver.

For every problem, follow this EXACT reasoning structure:

## Understanding
Restate the problem in your own words. Identify what is given and what is asked.

## Approach
List 2-3 possible approaches. Select the best one and explain why.

## Step-by-Step Solution
Number each step. Show your work for every calculation.

## Verification
Check your answer using a different method or by working backwards.

## Final Answer
State the answer clearly in one sentence.
"""

def structured_cot(problem):
    """Solve a problem using structured chain-of-thought."""
    messages = [
        {"role": "system", "content": STRUCTURED_COT_SYSTEM},
        {"role": "user", "content": problem}
    ]
    return call_llm(messages, temperature=0.0, max_tokens=2000)

# Example
result = structured_cot("""
A store offers a 25% discount on all items. After the discount,
there's an additional 10% off for loyalty members. Sales tax is 8%.
If a loyalty member buys an item originally priced at $200,
what's the final amount they pay?
""")
print(result)
```

**Why this is better than "think step by step":**
- You get a consistent output format every time
- The Verification section catches errors the model would otherwise miss
- You can parse the sections programmatically (split on `##`)
- Students and reviewers can follow the reasoning clearly

### 2.3 Programmatic CoT — Parsing and Validating Steps

In agentic workflows, the CoT output needs to be machine-readable. Here's how to make the model produce parseable reasoning:

```python
import json

COT_JSON_SYSTEM = """You are a calculation agent. For every math problem:

1. Break the problem into individual calculation steps.
2. Execute each step showing the operation and result.
3. Verify the final answer.

Respond ONLY with this JSON format:
{
    "steps": [
        {"step": 1, "description": "what this step does", "operation": "math expression", "result": <number>},
        {"step": 2, "description": "...", "operation": "...", "result": <number>}
    ],
    "verification": "description of how you verified",
    "final_answer": <number>,
    "unit": "string (e.g., dollars, km, items)"
}
"""

def validated_cot(problem):
    """Solve with CoT and validate each step programmatically."""
    messages = [
        {"role": "system", "content": COT_JSON_SYSTEM},
        {"role": "user", "content": problem}
    ]

    response = call_llm(messages, temperature=0.0)
    result = json.loads(response)

    # Validate: Check that steps are sequential
    for i, step in enumerate(result["steps"]):
        assert step["step"] == i + 1, f"Step numbering error at step {i+1}"
        assert isinstance(step["result"], (int, float)), f"Step {i+1} result is not a number"

    # Validate: Final answer should match the last step's result
    last_result = result["steps"][-1]["result"]
    if abs(result["final_answer"] - last_result) > 0.01:
        print(f"WARNING: Final answer {result['final_answer']} doesn't match last step {last_result}")

    return result

# Usage
answer = validated_cot("If a car travels at 65 mph for 3.5 hours, how far does it go?")
print(f"Answer: {answer['final_answer']} {answer['unit']}")
for step in answer["steps"]:
    print(f"  Step {step['step']}: {step['description']} → {step['result']}")
```

### 2.4 Auto-CoT: Generating Chain-of-Thought Examples Automatically

Instead of hand-crafting few-shot CoT examples, generate them automatically:

```python
def auto_cot(task_description, example_problems, target_problem):
    """Generate CoT demonstrations automatically, then solve the target problem."""

    # Step 1: Generate CoT for each example
    demonstrations = []
    for problem in example_problems:
        messages = [
            {"role": "user", "content": f"{problem}\n\nThink step by step and show your reasoning:"}
        ]
        reasoning = call_llm(messages, temperature=0.0)
        demonstrations.append(f"Problem: {problem}\nReasoning: {reasoning}\n")

    # Step 2: Use generated demonstrations as few-shot examples
    demo_text = "\n---\n".join(demonstrations)
    messages = [
        {"role": "user", "content": f"""{task_description}

Here are some example solutions:

{demo_text}

---

Now solve this problem the same way:
Problem: {target_problem}
Reasoning:"""}
    ]

    return call_llm(messages, temperature=0.0, max_tokens=1500)

# Usage
result = auto_cot(
    task_description="Solve word problems involving percentages and discounts.",
    example_problems=[
        "A shirt costs $40 and is 30% off. What's the sale price?",
        "A restaurant bill is $85. You tip 18%. What's the total?"
    ],
    target_problem="A laptop costs $1200. It's 15% off, and you have a $50 coupon applied after the discount. What do you pay?"
)
```

---

## 3. ReAct: Building Your First Agent Loop

### 3.1 The ReAct Pattern Explained

ReAct (Reason + Act) is the backbone of most modern AI agents. The model alternates between **thinking** (reasoning about what to do) and **acting** (calling tools), then **observing** the results.

```
User Question
     ↓
┌──────────────────┐
│  Thought: I need  │
│  to search for... │
├──────────────────┤
│  Action: search() │ ──→ External Tool ──→ Result
├──────────────────┤
│  Observation:     │
│  I found that...  │
├──────────────────┤
│  Thought: Now I   │
│  need to...       │
├──────────────────┤
│  Action: calc()   │ ──→ External Tool ──→ Result
├──────────────────┤
│  Observation:     │
│  The result is... │
├──────────────────┤
│  Thought: I have  │
│  all the info.    │
├──────────────────┤
│  Final Answer:    │
│  ...              │
└──────────────────┘
```

### 3.2 Implementing ReAct from Scratch

Let's build a complete ReAct agent with tool execution:

```python
import json
import re
import math

# ── Step 1: Define tools ─────────────────────────────────────────────

def search_web(query):
    """Simulate a web search (replace with real API in production)."""
    # In production, use SerpAPI, Tavily, or similar
    fake_results = {
        "population of france": "The population of France is approximately 68.4 million (2024).",
        "gdp of france": "France's GDP is approximately $3.05 trillion (2024).",
        "capital of france": "The capital of France is Paris.",
    }
    for key, value in fake_results.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"

def calculate(expression):
    """Safely evaluate a math expression."""
    try:
        # Only allow safe math operations
        allowed = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'sqrt': math.sqrt,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_current_date():
    """Return the current date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

# ── Step 2: Tool registry ────────────────────────────────────────────

TOOLS = {
    "search_web": {
        "function": search_web,
        "description": "Search the web for information. Input: a search query string.",
        "example": 'search_web("population of France 2024")'
    },
    "calculate": {
        "function": calculate,
        "description": "Evaluate a math expression. Input: a valid Python math expression.",
        "example": 'calculate("68400000 / 3050000000000 * 100")'
    },
    "get_current_date": {
        "function": get_current_date,
        "description": "Get today's date. Input: none needed.",
        "example": 'get_current_date()'
    },
}

# ── Step 3: ReAct system prompt ──────────────────────────────────────

def build_react_system_prompt(tools):
    """Build the system prompt with tool descriptions."""
    tool_descriptions = ""
    for name, info in tools.items():
        tool_descriptions += f"- {name}: {info['description']}\n  Example: {info['example']}\n"

    return f"""You are a helpful assistant that reasons step-by-step and uses tools when needed.

## Available Tools
{tool_descriptions}

## How to Respond
For each step, use this EXACT format:

Thought: [your reasoning about what to do next]
Action: [tool_name("input")]
<PAUSE>

After receiving the observation, continue with another Thought/Action or provide the final answer:

Thought: [reasoning based on observation]
Final Answer: [your complete answer to the user's question]

## Rules
- Always start with a Thought
- Use tools when you need external information or calculations
- Never guess when you can look something up
- After receiving enough information, provide a Final Answer
- Maximum 5 tool calls per question
"""

# ── Step 4: The agent loop ───────────────────────────────────────────

def react_agent(user_question, max_steps=5, verbose=True):
    """Run a ReAct agent loop."""
    system_prompt = build_react_system_prompt(TOOLS)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    for step in range(max_steps):
        # Get the model's next thought + action
        response = call_llm(messages, temperature=0.0, max_tokens=500)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Step {step + 1}:")
            print(response)

        # Check if we have a final answer
        if "Final Answer:" in response:
            # Extract the final answer
            final = response.split("Final Answer:")[-1].strip()
            return final

        # Parse the action
        action_match = re.search(r'Action:\s*(\w+)\("?([^"]*)"?\)', response)
        if not action_match:
            # Model didn't call a tool — add response and ask it to continue
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please continue with an Action or provide a Final Answer."})
            continue

        tool_name = action_match.group(1)
        tool_input = action_match.group(2)

        # Execute the tool
        if tool_name in TOOLS:
            observation = TOOLS[tool_name]["function"](tool_input)
        else:
            observation = f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOLS.keys())}"

        if verbose:
            print(f"\nObservation: {observation}")

        # Add the exchange to the conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Agent reached maximum steps without a final answer."

# ── Step 5: Run the agent ────────────────────────────────────────────

answer = react_agent("What is the GDP per capita of France? Show me the calculation.")
print(f"\n{'='*60}")
print(f"FINAL ANSWER: {answer}")
```

### 3.3 Handling Errors in the Agent Loop

Real agents hit errors. The agent needs to know how to recover:

```python
REACT_WITH_RECOVERY = """You are a research agent with tools.

## Error Recovery Rules
- If a tool returns an error, try a different approach or rephrase the input
- If a search returns no results, try different search terms
- If a calculation fails, check the expression syntax and retry
- If you've retried 2 times and still failing, explain what went wrong in your Final Answer
- NEVER make up information — if you can't find it, say so

## Example of Error Recovery
Thought: I need to find the population of Xyzland.
Action: search_web("population of Xyzland")
Observation: No results found for: population of Xyzland
Thought: The search didn't find results. Let me try with different terms.
Action: search_web("Xyzland country population census")
Observation: No results found for: Xyzland country population census
Thought: I've tried twice and can't find this information. I should be honest about this.
Final Answer: I wasn't able to find the population of Xyzland. This may not be a real country, or the data may not be available in my search results.
"""
```

### 3.4 ReAct with OpenAI Function Calling (The Modern Way)

The text-parsing approach above is educational, but production agents use OpenAI's built-in function calling:

```python
# Define tools in OpenAI's function calling format
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

def react_agent_with_function_calling(user_question, max_steps=5):
    """ReAct agent using OpenAI's native function calling."""
    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Think step by step."},
        {"role": "user", "content": user_question}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Let the model decide when to use tools
            temperature=0.0,
        )

        msg = response.choices[0].message
        messages.append(msg)

        # If no tool call, the model is giving the final answer
        if not msg.tool_calls:
            return msg.content

        # Execute each tool call
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            # Execute the tool
            if func_name == "search_web":
                result = search_web(func_args["query"])
            elif func_name == "calculate":
                result = calculate(func_args["expression"])
            else:
                result = f"Unknown tool: {func_name}"

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Max steps reached."
```

**Why function calling is better:**
- No fragile text parsing (regex) needed
- The model produces structured JSON for tool inputs
- Tool results are properly attributed in the conversation
- Works with OpenAI's parallel function calling (multiple tools at once)

---

## 4. Tree-of-Thought: Multi-Path Reasoning

### 4.1 When CoT Isn't Enough

Chain-of-Thought follows a single path. But some problems have multiple valid approaches, and the first path you try might be a dead end. Tree-of-Thought (ToT) explores multiple reasoning paths and evaluates which is most promising.

```
                    Problem
                   /   |   \
              Path A  Path B  Path C
              /  \      |      /  \
           A1   A2     B1    C1   C2
           ↓          ↓          ↓
        (dead end)  (promising)  (solution!)
```

### 4.2 Implementing Tree-of-Thought

```python
def tree_of_thought(problem, n_paths=3, temperature=0.7):
    """
    Explore multiple reasoning paths for a problem.

    Steps:
    1. Generate N different approaches
    2. Evaluate each approach
    3. Pursue the most promising one to completion
    """

    # ── Step 1: Generate multiple approaches ─────────────────────────
    generation_prompt = f"""Given this problem, propose {n_paths} distinctly different approaches
to solving it. For each approach, write 2-3 sentences describing the strategy.

Problem: {problem}

Respond as JSON:
{{
    "approaches": [
        {{"id": 1, "strategy": "description of approach 1", "first_step": "what you'd do first"}},
        {{"id": 2, "strategy": "description of approach 2", "first_step": "what you'd do first"}},
        {{"id": 3, "strategy": "description of approach 3", "first_step": "what you'd do first"}}
    ]
}}"""

    approaches_raw = call_llm(
        [{"role": "user", "content": generation_prompt}],
        temperature=temperature
    )
    approaches = json.loads(approaches_raw)

    # ── Step 2: Evaluate each approach ───────────────────────────────
    eval_prompt = f"""You are evaluating different approaches to solve a problem.

Problem: {problem}

Approaches:
{json.dumps(approaches['approaches'], indent=2)}

For each approach, rate it on:
1. Feasibility (1-5): Can this approach actually work?
2. Efficiency (1-5): How direct is this path to the solution?
3. Correctness risk (1-5): How likely is this to produce a correct answer? (5 = very likely)

Respond as JSON:
{{
    "evaluations": [
        {{"id": 1, "feasibility": N, "efficiency": N, "correctness": N, "total": N, "reasoning": "..."}},
        {{"id": 2, "feasibility": N, "efficiency": N, "correctness": N, "total": N, "reasoning": "..."}},
        {{"id": 3, "feasibility": N, "efficiency": N, "correctness": N, "total": N, "reasoning": "..."}}
    ],
    "best_approach_id": N
}}"""

    evaluation_raw = call_llm(
        [{"role": "user", "content": eval_prompt}],
        temperature=0.0  # Deterministic evaluation
    )
    evaluation = json.loads(evaluation_raw)
    best_id = evaluation["best_approach_id"]

    # ── Step 3: Execute the best approach ────────────────────────────
    best_approach = next(a for a in approaches["approaches"] if a["id"] == best_id)

    execution_prompt = f"""Solve this problem using the specified approach.

Problem: {problem}

Approach: {best_approach['strategy']}

Solve it step by step following this approach. Be thorough and show all work.
End with a clear Final Answer.
"""

    solution = call_llm(
        [{"role": "user", "content": execution_prompt}],
        temperature=0.0,
        max_tokens=2000
    )

    return {
        "approaches": approaches["approaches"],
        "evaluation": evaluation["evaluations"],
        "best_approach": best_approach,
        "solution": solution
    }

# Usage
result = tree_of_thought("""
A company needs to reduce its cloud computing costs by 30% while maintaining
the same level of service. They currently spend $50,000/month on AWS with
the following breakdown:
- EC2 instances: $25,000 (mix of on-demand and some reserved)
- S3 storage: $8,000
- RDS databases: $10,000
- Data transfer: $4,000
- Other services: $3,000

What strategy should they adopt?
""")

print(f"Best approach: {result['best_approach']['strategy']}")
print(f"\nSolution:\n{result['solution']}")
```

### 4.3 Breadth-First vs Depth-First ToT

Two strategies for exploring the tree:

```python
def tot_breadth_first(problem, n_branches=3, depth=2):
    """
    Breadth-first: Generate all branches at each level, evaluate, keep top ones.
    Good for: Problems where early decisions are critical.
    """
    current_paths = [{"steps": [], "partial_solution": "Start"}]

    for level in range(depth):
        all_candidates = []

        for path in current_paths:
            # Generate branches from this path
            branch_prompt = f"""Problem: {problem}

Steps taken so far: {json.dumps(path['steps'])}

Propose {n_branches} different next steps. Each should be a distinct approach.
Respond as JSON: {{"branches": ["step description 1", "step description 2", "step description 3"]}}"""

            branches_raw = call_llm(
                [{"role": "user", "content": branch_prompt}],
                temperature=0.7
            )
            branches = json.loads(branches_raw)

            for branch in branches["branches"]:
                all_candidates.append({
                    "steps": path["steps"] + [branch],
                    "partial_solution": branch
                })

        # Evaluate all candidates, keep top N
        eval_prompt = f"""Problem: {problem}

Rate each partial solution path (1-10) for how promising it looks:
{json.dumps([{"id": i, "steps": c["steps"]} for i, c in enumerate(all_candidates)], indent=2)}

Respond as JSON: {{"ratings": [{{"id": N, "score": N}}]}}"""

        ratings_raw = call_llm(
            [{"role": "user", "content": eval_prompt}],
            temperature=0.0
        )
        ratings = json.loads(ratings_raw)

        # Keep top paths
        scored = [(r["score"], all_candidates[r["id"]]) for r in ratings["ratings"]]
        scored.sort(key=lambda x: x[0], reverse=True)
        current_paths = [path for _, path in scored[:n_branches]]

    return current_paths[0]  # Return the best path
```

---

## 5. Graph-of-Thought: Non-Linear Problem Solving

### 5.1 Why Graphs?

Trees assume reasoning branches don't interact. But in real problems, insights from one branch can inform another. Graph-of-Thought (GoT) allows:

- **Merging**: Combine partial solutions from different branches
- **Looping**: Refine a thought based on later insights
- **Dependencies**: One thought explicitly depends on another

```
     Thought A ──→ Thought B ──→ Thought D
         ↓              ↓            ↗
     Thought C ─────────┘       Merge
         ↓                        ↗
     Thought E ──────────────────┘
```

### 5.2 Implementing Graph-of-Thought

```python
class ThoughtNode:
    """A single node in the thought graph."""
    def __init__(self, thought_id, content, node_type="thought"):
        self.id = thought_id
        self.content = content
        self.type = node_type  # "thought", "merge", "refine"
        self.children = []
        self.parents = []
        self.score = None

    def __repr__(self):
        return f"Node({self.id}: {self.content[:50]}...)"


class GraphOfThought:
    """Implement Graph-of-Thought reasoning."""

    def __init__(self):
        self.nodes = {}
        self.counter = 0

    def add_thought(self, content, parent_ids=None, node_type="thought"):
        """Add a thought node to the graph."""
        self.counter += 1
        node = ThoughtNode(self.counter, content, node_type)
        self.nodes[self.counter] = node

        if parent_ids:
            for pid in parent_ids:
                parent = self.nodes[pid]
                parent.children.append(node)
                node.parents.append(parent)

        return node

    def generate_thoughts(self, problem, n=3):
        """Generate initial diverse thoughts about the problem."""
        prompt = f"""Break this problem into {n} independent sub-problems that can be solved separately.

Problem: {problem}

Respond as JSON:
{{"sub_problems": ["sub-problem 1 description", "sub-problem 2", "sub-problem 3"]}}"""

        result = json.loads(call_llm([{"role": "user", "content": prompt}]))

        nodes = []
        for sub in result["sub_problems"]:
            node = self.add_thought(sub)
            nodes.append(node)

        return nodes

    def solve_thought(self, node):
        """Solve an individual thought node."""
        # Gather context from parent nodes
        parent_context = ""
        if node.parents:
            parent_solutions = [f"- {p.content}" for p in node.parents if p.score]
            if parent_solutions:
                parent_context = f"\n\nContext from previous reasoning:\n" + "\n".join(parent_solutions)

        prompt = f"""Solve this sub-problem concisely.{parent_context}

Sub-problem: {node.content}

Respond as JSON:
{{"solution": "your solution", "confidence": 0.0-1.0, "key_insight": "main takeaway"}}"""

        result = json.loads(call_llm([{"role": "user", "content": prompt}]))
        node.content = f"{node.content}\nSolution: {result['solution']}"
        node.score = result["confidence"]
        return result

    def merge_thoughts(self, node_ids, merge_instruction):
        """Merge insights from multiple nodes into a new node."""
        sources = [self.nodes[nid] for nid in node_ids]
        source_text = "\n".join([f"Insight {n.id}: {n.content}" for n in sources])

        prompt = f"""Merge these insights into a unified solution.

{source_text}

Merge instruction: {merge_instruction}

Respond as JSON:
{{"merged_solution": "unified answer", "confidence": 0.0-1.0}}"""

        result = json.loads(call_llm([{"role": "user", "content": prompt}]))
        merged_node = self.add_thought(
            result["merged_solution"],
            parent_ids=node_ids,
            node_type="merge"
        )
        merged_node.score = result["confidence"]
        return merged_node

    def refine_thought(self, node_id, feedback):
        """Refine a thought based on feedback or new information."""
        node = self.nodes[node_id]

        prompt = f"""Refine this solution based on feedback.

Original solution: {node.content}
Feedback: {feedback}

Respond as JSON:
{{"refined_solution": "improved answer", "changes_made": "what changed", "confidence": 0.0-1.0}}"""

        result = json.loads(call_llm([{"role": "user", "content": prompt}]))
        refined_node = self.add_thought(
            result["refined_solution"],
            parent_ids=[node_id],
            node_type="refine"
        )
        refined_node.score = result["confidence"]
        return refined_node


# ── Using the Graph-of-Thought ───────────────────────────────────────

def solve_with_got(problem):
    """Solve a complex problem using Graph-of-Thought."""
    got = GraphOfThought()

    # Step 1: Decompose into sub-problems
    print("Step 1: Decomposing problem...")
    initial_thoughts = got.generate_thoughts(problem, n=3)

    # Step 2: Solve each sub-problem independently
    print("Step 2: Solving sub-problems...")
    solutions = []
    for node in initial_thoughts:
        result = got.solve_thought(node)
        print(f"  Node {node.id}: confidence={result['confidence']}")
        solutions.append(result)

    # Step 3: Merge solutions
    print("Step 3: Merging insights...")
    node_ids = [n.id for n in initial_thoughts]
    merged = got.merge_thoughts(
        node_ids,
        "Combine all sub-problem solutions into a complete, coherent answer."
    )

    # Step 4: Refine if confidence is low
    if merged.score and merged.score < 0.8:
        print("Step 4: Refining (confidence was low)...")
        refined = got.refine_thought(
            merged.id,
            "Double-check for consistency between the merged parts. Fix any contradictions."
        )
        return refined.content

    return merged.content

# Example
result = solve_with_got("""
Design a notification system for a food delivery app that handles:
- Order status updates (placed, preparing, out for delivery, delivered)
- Promotional offers targeted by user preferences
- Driver communication (delayed, can't find address)
Each type has different urgency levels and delivery channels (push, SMS, email).
""")
print(f"\nFinal solution:\n{result}")
```

### 5.3 GoT vs ToT vs CoT — When Each Shines

| Aspect | CoT | ToT | GoT |
|--------|-----|-----|-----|
| **Structure** | Linear chain | Branching tree | Arbitrary graph |
| **Exploration** | Single path | Multiple paths, pick best | Multiple paths, merge results |
| **Best for** | Step-by-step reasoning | Problems with multiple valid approaches | Complex problems with interacting sub-problems |
| **Cost (API calls)** | 1 call | 3-10+ calls | 5-15+ calls |
| **Complexity** | Low | Medium | High |
| **Example** | Math problems, logic | Strategy/design decisions | System design, research synthesis |

---

## 6. Choosing the Right Reasoning Pattern

### 6.1 The Decision Framework

```
Is the problem straightforward with a clear solution path?
  └─ YES → Chain-of-Thought (CoT)
  └─ NO → Are there multiple valid approaches you want to compare?
            └─ YES → Do the sub-problems interact with each other?
                       └─ YES → Graph-of-Thought (GoT)
                       └─ NO  → Tree-of-Thought (ToT)
            └─ NO → Does the problem require external tools/data?
                      └─ YES → ReAct
                      └─ NO  → Structured CoT with verification
```

### 6.2 Pattern Selection by Task Type

| Task | Best Pattern | Why |
|------|-------------|-----|
| Math word problems | Structured CoT | Clear step-by-step, easy to verify |
| Code debugging | ReAct + CoT | Reason about the bug, run tests to verify |
| Business strategy | ToT | Multiple valid approaches, need to evaluate trade-offs |
| Research synthesis | GoT | Multiple sources, insights need to be merged |
| Customer support routing | ReAct | Need to look up order info, check policies |
| System architecture design | GoT | Sub-components interact and constrain each other |
| Data extraction | CoT (simple) or ReAct (complex) | Depends on whether tools are needed |
| Creative writing | ToT | Generate multiple drafts, pick the best |

---

## 7. Composing Patterns: Real-World Combinations

In practice, you rarely use just one pattern. Here's how they combine:

### 7.1 ReAct + CoT: The Reasoning Agent

```python
REASONING_AGENT_SYSTEM = """You are a research agent that thinks carefully before acting.

## Approach
For every question:
1. THINK: Break the problem down (Chain-of-Thought)
2. PLAN: Identify what information you need and which tools to use
3. ACT: Execute tools one at a time
4. VERIFY: Check if the answer makes sense

## Format
Thought: [Chain-of-Thought reasoning — break the problem into steps]
Plan: [numbered list of what you need to do]
Action: [tool_call]
<PAUSE>
Observation: [tool result]
Thought: [reasoning about the result]
... (continue until done)
Verification: [sanity check your answer]
Final Answer: [complete answer]
"""
```

### 7.2 ToT + ReAct: The Strategic Agent

```python
def strategic_agent(problem):
    """
    1. Use ToT to generate and evaluate strategies
    2. Use ReAct to execute the best strategy
    """
    # Phase 1: Strategic planning (ToT)
    strategies = tree_of_thought(problem, n_paths=3)
    best_strategy = strategies["best_approach"]["strategy"]

    # Phase 2: Execution (ReAct)
    execution_question = f"""Execute this strategy step by step:

Original problem: {problem}
Strategy to follow: {best_strategy}

Use your tools to gather information and calculate as needed."""

    result = react_agent(execution_question)
    return result
```

---

## 8. Hands-On Exercises

### Exercise 1: Build a Structured CoT Problem Solver

**Goal:** Create a structured chain-of-thought solver that breaks problems into verifiable steps.

**What you'll build:**
- A function that takes a word problem and returns a JSON object with numbered steps, each with an operation and result
- A validation function that checks each step's math

**Starter code:**

```python
import json
from openai import OpenAI

client = OpenAI()

# ── YOUR TASK: Complete the system prompt ────────────────────────────
# The system prompt should instruct the model to:
# 1. Restate the problem
# 2. Break it into numbered steps with explicit operations
# 3. Provide a verification step
# 4. Return everything as valid JSON

COT_SOLVER_SYSTEM = """
# TODO: Write your system prompt here.
# It should produce JSON output with this structure:
# {
#     "restatement": "...",
#     "steps": [
#         {"step": 1, "description": "...", "operation": "...", "result": ...},
#         ...
#     ],
#     "verification": "...",
#     "final_answer": ...,
#     "unit": "..."
# }
"""

def solve_problem(problem):
    """Solve a word problem using structured CoT."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": COT_SOLVER_SYSTEM},
            {"role": "user", "content": problem}
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def validate_solution(solution):
    """Validate that the solution steps are consistent."""
    errors = []

    # TODO: Check that steps are numbered sequentially
    # TODO: Check that the final_answer matches the last step's result
    # TODO: Check that all result values are numbers

    return errors


# ── Test problems ────────────────────────────────────────────────────
test_problems = [
    "A bakery makes 240 cupcakes. They sell 35% in the morning and 40% of the remaining in the afternoon. How many cupcakes are left?",
    "A train travels from City A to City B at 80 km/h and returns at 60 km/h. If the distance is 240 km, what is the average speed for the round trip?",
    "A rectangle's length is 3 times its width. If the perimeter is 96 cm, find the area.",
]

for i, problem in enumerate(test_problems, 1):
    print(f"\n{'='*60}")
    print(f"Problem {i}: {problem}")

    solution = solve_problem(problem)
    errors = validate_solution(solution)

    print(f"Answer: {solution['final_answer']} {solution.get('unit', '')}")
    print(f"Steps: {len(solution['steps'])}")

    if errors:
        print(f"VALIDATION ERRORS: {errors}")
    else:
        print("Validation: PASSED")
```

**Expected outcome:** Students should be able to solve all 3 problems with their CoT solver and have the validation pass. The key learning is that you can make LLM reasoning verifiable by demanding structured output.

---

### Exercise 2: Build a ReAct Agent with Tool Use

**Goal:** Build a working ReAct agent that uses tools to answer questions it can't answer from memory alone.

**What you'll build:**
- A ReAct agent with a search tool and a calculator tool
- The agent loop that parses thoughts, executes actions, and arrives at a final answer

**Starter code:**

```python
import json
import re
from openai import OpenAI

client = OpenAI()

# ── Tools (these are provided) ───────────────────────────────────────

KNOWLEDGE_BASE = {
    "paris": {"population": 2161000, "country": "France", "area_km2": 105.4},
    "london": {"population": 8982000, "country": "UK", "area_km2": 1572},
    "tokyo": {"population": 13960000, "country": "Japan", "area_km2": 2194},
    "new york": {"population": 8336000, "country": "USA", "area_km2": 783.8},
    "mumbai": {"population": 12442000, "country": "India", "area_km2": 603.4},
}

def lookup_city(city_name):
    """Look up information about a city."""
    city = city_name.lower().strip()
    if city in KNOWLEDGE_BASE:
        data = KNOWLEDGE_BASE[city]
        return json.dumps(data)
    return f"City '{city_name}' not found. Available cities: {', '.join(KNOWLEDGE_BASE.keys())}"

def calculate(expression):
    """Evaluate a math expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"round": round, "abs": abs})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "lookup_city": {
        "function": lookup_city,
        "description": "Look up population, country, and area for a city. Input: city name.",
        "example": 'lookup_city("Paris")'
    },
    "calculate": {
        "function": calculate,
        "description": "Evaluate a math expression. Input: Python math expression.",
        "example": 'calculate("8982000 / 1572")'
    }
}

# ── YOUR TASK: Build the agent ───────────────────────────────────────

# TODO 1: Write the system prompt
# - List available tools with descriptions and examples
# - Define the Thought / Action / Observation format
# - Include rules for error handling

REACT_SYSTEM = """
# TODO: Write your system prompt here
"""

# TODO 2: Implement the agent loop
def my_react_agent(question, max_steps=5):
    """
    Implement the ReAct loop:
    1. Send the question to the model
    2. If response contains "Final Answer:" → return it
    3. If response contains "Action:" → parse and execute the tool
    4. Add the observation to the conversation
    5. Repeat
    """
    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        # TODO: Get model response
        # TODO: Check for Final Answer
        # TODO: Parse action and execute tool
        # TODO: Add observation to messages
        pass

    return "Max steps reached without an answer."


# ── Test questions ───────────────────────────────────────────────────
test_questions = [
    "What is the population density of Tokyo? (people per square km)",
    "Which city is more densely populated: London or Mumbai? Show the calculations.",
    "What is the combined population of all European cities in the database?",
]

for q in test_questions:
    print(f"\nQ: {q}")
    answer = my_react_agent(q)
    print(f"A: {answer}")
```

**Expected outcome:** Students build a working agent that can look up city data and do calculations. The key learning is understanding the agent loop — how the conversation grows with each Thought/Action/Observation cycle.

**Hints if students get stuck:**
1. The system prompt should include the exact Thought/Action/Observation format
2. Use `re.search(r'Action:\s*(\w+)\("([^"]*)"\)', response)` to parse actions
3. Add the assistant's response AND the observation as separate messages
4. The model needs to see `Observation: <result>` to continue reasoning

---

## 9. Quick Reference

### Pattern Cheat Sheet

```
Chain-of-Thought (CoT)
  When: Step-by-step reasoning, math, logic
  Cost: 1 API call
  Template: "Think step by step" OR structured JSON output

ReAct (Reason + Act)
  When: Need external tools or data
  Cost: 2-10 API calls (depends on tool calls needed)
  Template: Thought → Action → Observation → repeat → Final Answer

Tree-of-Thought (ToT)
  When: Multiple valid approaches, need to compare
  Cost: 3-10+ API calls
  Template: Generate paths → Evaluate → Pursue best → Solve

Graph-of-Thought (GoT)
  When: Complex problems with interacting sub-problems
  Cost: 5-15+ API calls
  Template: Decompose → Solve independently → Merge → Refine
```

### Implementation Decision Guide

```
Need to solve it in 1 API call?
  → Structured CoT with JSON output

Need tools (search, APIs, databases)?
  → ReAct (with OpenAI function calling for production)

Need to compare multiple strategies?
  → Tree-of-Thought

Need to synthesize from multiple independent analyses?
  → Graph-of-Thought

Building a production agent?
  → ReAct with function calling + CoT in the system message
```

### Common Pitfalls

```
❌ Using ToT/GoT for simple problems    → Overkill, just use CoT
❌ Not setting temperature=0 for routing → Non-deterministic behavior
❌ Parsing tool calls with regex in prod → Use function calling instead
❌ No max_steps in agent loop            → Infinite loops and runaway costs
❌ Forgetting error recovery in ReAct    → Agent crashes on first tool error
❌ Using GoT without a merge step        → Just disconnected analyses
```

---

> **Next Session Preview:** Session 8 covers **Self-Reflection & Critique** — building agents that evaluate their own outputs and improve them automatically. You'll implement the Reflexion algorithm and learn how to build AutoGen-style evaluators.

---

*© BIA® School of Technology & AI — Generative AI & Agentic AI Development Program*
