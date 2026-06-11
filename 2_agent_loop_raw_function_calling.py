from dotenv import load_dotenv
# from _typeshed import OpenBinaryMode
load_dotenv()

import ollama
import sys

from langsmith import traceable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MAX_ITERATIONS = 10
# MODEL = "qwen2.5:latest"
MODEL = "gemma4:e4b"


# Tools
@traceable(run_type="tool")
def get_product_price(product:str) -> float:
    """ Look up the price of a product in the database """
    print(f"Looking up the price of {product}")
    prices = {"laptop": 1299.99, "phone": 999.99, "tablet": 699.99}
    return prices.get(product, "Product not found")

@traceable(run_type="tool")
def apply_discount(price:float, discount_tier:str) -> float:
    """ Apply a discount to a price based on a discount tier \
        Available discount tiers: "gold", "silver", "bronze" """

    print(f"Applying discount {discount_tier} to price {price}")
    discount_percentages = {"bronze": 10, "silver": 15, "gold": 20}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must manually define the JSON schema for each function.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The name of the product to look up"
                    }
                },
                "required": ["product"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to a price based on a discount tier",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The price to apply the discount to"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply",
                        "enum": ["bronze", "silver", "gold"]
                    }
                },
                "required": ["price", "discount_tier"],
            },
        }
    }
]


# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace the Ollama call for LangSmith.
@traceable(name="Traced Ollama Call", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# Agent loop
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("="*60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that can use the following tools to answer questions. "
                "Strict rules: \n"
                "1. You must use the tools provided to you.\n"
                "2. You must answer the question in a concise manner.\n"
                "3. You must answer the question in a friendly manner.\n"
                "4. You must answer the question in a helpful manner.\n"
                "5. You must answer the question in a concise manner.\n"
                "6. You must answer the question in a friendly manner.\n"
            )
        },
        {"role": "user", "content": question},    
    ]
    
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"---Iteration {i} ---")

        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages)
        ai_message=response.message

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content

        messages.append(ai_message.model_dump(exclude_none=True))

        for tool_call in tool_calls:
            # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            print(f"Calling tool: {tool_name} with args: {tool_args}")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool {tool_name} not found")

            # Difference 7: Direct function call instead of tool_to_use.invoke()
            observation = tool_to_use(**tool_args)

            print(f"[Tool Result]: {observation}")

            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(observation),
                }
            )

    print(f"Maximum number of iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent(input("Enter a question: "))
