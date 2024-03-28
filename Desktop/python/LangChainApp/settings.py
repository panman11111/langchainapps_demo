from langchain.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain import LLMMathChain
import json

# 定数
OPENAI_API_KEY = ""

# jsonからAPIキーを取得
with open('API_KEY.json', 'r') as f:
    OPENAI_API_KEY = json.load(f)["openai_api_key"]
    f.close()

# LLM、エージェントの構築
search = DuckDuckGoSearchRun()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.7, streaming=True)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name = "ddg-search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
    # Tool(
    #     name="",
    #     func=,
    #     description=""
    # ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
)