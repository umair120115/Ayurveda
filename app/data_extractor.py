from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

GROQ_API_TOKEN = "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"
GROQ_API_TOKEN = os.getenv("GROQ_API_TOKEN")
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192",streaming=True)

tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]

prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (provide the raw information gathered)

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
prompt = PromptTemplate.from_template(prompt_template)

agent=create_react_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# topic_to_search=
topic_to_research = "Different herbal medicines for different diseases having authenticated scientific evidence."
# search_results = agent_executor.invoke({"input": topic_to_research})
search_results = agent_executor.invoke({"input": topic_to_research})
collected_data = search_results['output']


# --- Data Processing and VectorDB Creation ---

if collected_data:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100 
    )
    documents = text_splitter.create_documents([collected_data])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")