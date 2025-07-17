from langchain_core.messages.ai import AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

import os
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


openai_llm = ChatOpenAI(model="gpt-4o-min")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")

search_tool = TavilySearch(max_results=1)


mentor_role = "Act as a mentor for someone who is preparing for interviews in the field of Frontend"

agent = create_react_agent(
    model=groq_llm,
    tools=[search_tool],
    # system_message=mentor_role
)

query = "I want to give a fresher level React interview"
state = {
    "messages": [
        # Optional if not passed in `system_message`
        SystemMessage(content=mentor_role),
        HumanMessage(content=query)
    ]
}
response = agent.invoke(state)
result_messages = response.get("messages")
ai_results=[message.content for message in result_messages if isinstance(message, AIMessage)]
print("AI RESULT ----> ", ai_results[-1])
print("ANOTHER ONE ---> ",ai_results)


# for msg in ai_results:
#     print(f"{msg} ----- \n\n")