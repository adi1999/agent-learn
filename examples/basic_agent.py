"""Basic example: trace an OpenAI agent and run the learning cycle.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/basic_agent.py
"""

from agentlearn import Engine

# Create the engine (stores knowledge in ./knowledge/)
engine = Engine(model="gpt-4o-mini")


@engine.trace
def my_agent(task_input: str) -> str:
    """A simple agent that answers questions using OpenAI."""
    from openai import OpenAI

    client = OpenAI()

    # Pull learned knowledge (returns "" if nothing relevant)
    knowledge = engine.get_knowledge(task_input)

    system_prompt = "You are a helpful assistant. Answer concisely."
    if knowledge:
        system_prompt += f"\n\n{knowledge}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_input},
        ],
    )
    return response.choices[0].message.content


def main():
    # Run the agent on several tasks to collect traces
    tasks = [
        "What is compound interest on $1000 at 5% for 3 years?",
        "Convert 100 USD to EUR at today's rate.",
        "What is the derivative of x^3 + 2x?",
        "Write a Python function to check if a string is a palindrome.",
        "Explain the difference between TCP and UDP in one paragraph.",
    ]

    print("Running agent on tasks...\n")
    for task in tasks:
        try:
            result = my_agent(task)
            print(f"Task: {task}")
            print(f"Result: {result[:100]}...")
            print()
        except Exception as e:
            print(f"Task: {task}")
            print(f"Error: {e}")
            print()

    # Check status
    status = engine.status()
    print(f"Traces collected: {status.traces_total}")
    print(f"Success: {status.traces_success}, Failure: {status.traces_failure}")
    print()

    # Run the learning cycle
    print("Running learning cycle...")
    report = engine.learn()
    print(f"Traces analyzed: {report.traces_analyzed}")
    print(f"Candidates generated: {report.candidates_generated}")
    print(f"Promoted: {report.candidates_promoted}")
    print(f"Cost: ${report.total_cost_usd:.4f}")


if __name__ == "__main__":
    main()
