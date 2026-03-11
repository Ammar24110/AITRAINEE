from Phase2.Agents.orchestrator_agent import OrchestratorAgent


def run_tests():
    """Simple manual tests for the multi-agent system."""

    orchestrator = OrchestratorAgent()

    tests = [
        "Add a new task for Ahmed: review the sales report before Friday",
        "What is the company policy for sick leave?",
        "List all pending tasks for Ahmed"
    ]

    for query in tests:
        print("\n==============================")
        print(f"User: {query}")

        response = orchestrator.handle_request(query)

        print("Response:", response.message)
        print("==============================\n")


if __name__ == "__main__":
    run_tests()