from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemma4:e4b"

# Tools
@tool
def get_product_price(product: str) -> float:
    """Look up the price of the product in the database."""

    print(f"Looking up the price of {product}")
    prices = {"Laptop": 999.99, "Phone": 699.99, "Tablet": 499.99}
    return prices.get(product, f"{product} not found")


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to the price based on the discount tier."""
    print(f"Applying discount {discount_tier} to price {price}")
    discount_percentages = {"None": 0, "Bronze": 10, "Silver": 15, "Gold": 20}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)


# Agent loop
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    llm = init_chat_model(f"ollama: {MODEL}", temperature=0)
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print(f"\n{'='*60}\n")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that can use the following tools to answer questions. "
                "Strict rules: \n"
                "1. You must use the tools provided to you.\n"
                "2. You must answer the question in a concise manner.\n"
                "3. You must answer the question in a friendly manner.\n"
                "4. You must answer the question in a helpful manner.\n"
                "5. You must answer the question in a concise manner.\n"
                "6. You must answer the question in a friendly manner.\n"
                "7. You must answer the question in a helpful manner.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"---Iteration {i}---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Call]: {tool_name} with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result]: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))

    print(f"Maximum number of iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    result = run_agent(input("Enter a question: "))
    print(f"\n{'='*60}\n")
    print(f"Result: {result}")