import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent import agent_app


def format_context_prompt(data):
    """
    Merges user input with the system context (Order details).
    """
    ctx = data["context"]
    prompt = f"""
    SYSTEM DATA (Hidden from user, visible to Agent):
    - Order placed at: {ctx["time_placed"]}
    - Items: {ctx["items"]}
    - ETA: {ctx["eta"]}
    - Status: {ctx["status"]}

    USER COMPLAINT/QUERY:
    "{data["user_input"]}"
    """
    return prompt


async def run_tests():
    # Load Data
    with open("test_data.json", "r") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases.\n")

    for i, case in enumerate(test_cases):
        print("=" * 60)
        print(f"TEST CASE #{case['id']}")
        print(f"INPUT: {case['user_input']}")
        print(
            f"CTX:   Status: {case['context']['status']} | Items: {case['context']['items']}"
        )
        print("-" * 20)

        # Create the initial message state
        # We give the LLM a system instruction + the specific case context
        system_instruction = SystemMessage(
            content="""
        You are ZomaBot, an automated support agent.
        Use the provided SYSTEM DATA to validate the user's claims.
        - If items are missing/bad, use 'process_refund'.
        - If safety/tech issue, use 'escalate_to_support_admin'.
        - If delivery instruction/location, use 'contact_delivery_partner'.
        - Otherwise, answer politely.
        """
        )

        user_msg = HumanMessage(content=format_context_prompt(case))

        messages = [system_instruction, user_msg]

        # Run Agent
        # Recursion limit prevents infinite loops if agent gets confused
        final_state = await agent_app.ainvoke(
            {"messages": messages}, config={"recursion_limit": 10}
        )

        # Analyze Results
        history = final_state["messages"]

        # Find if any tool was called
        tool_calls = []
        for msg in history:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc["name"])

        bot_response = history[-1].content

        # Output result
        if tool_calls:
            print(f"ACTION TAKEN: {', '.join(tool_calls)}")
        else:
            print("ACTION TAKEN: None (Direct Reply)")

        print(f"RESPONSE:     {bot_response}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
