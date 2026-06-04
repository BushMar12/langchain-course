from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent to source candidates"""

    url:str = Field(description="The URL of the job posting")

class AgentResponse(BaseModel):
    """Schema for the response from the agent"""

    answer:str = Field(description="The agent's answer to the user's question")
    sources:List[Source] = Field(default_factory=list, description="The sources used by the agent to answer the user's question")

llm = ChatOllama(model="qwen2.5:latest", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from react-agent!")

    response = agent.invoke({"messages": [HumanMessage(content="Search for 3 job postings for an AI engineer using Langchain in the bay area on LinkedIn and list their details")]})

    print(response)

if __name__ == "__main__":
    main()
