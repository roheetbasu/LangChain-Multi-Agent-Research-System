from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

#Model Initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.0,
    max_retries=2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

#1st Agent: Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        
    )

#2nd Agent: Reader Agent
def build_read_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        
    )
    
    