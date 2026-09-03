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


# writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.
     
Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research critic and fact-checker. "
        "Your job is to carefully evaluate research reports for accuracy, "
        "completeness, clarity, source quality, and logical consistency."
    ),
    (
        "human",
        """Critically review the research report below.

Topic:
{topic}

Research Gathered:
{research}

Research Report:
{report}

Evaluate the report using the following criteria:

1. Accuracy
   - Identify claims that appear unsupported or potentially incorrect.
   - Check whether the report stays consistent with the research provided.

2. Completeness
   - Identify important information from the research that was missed.
   - Check whether the report adequately addresses the topic.

3. Key Findings
   - Check whether there are at least 3 meaningful and well-explained findings.
   - Identify findings that are vague, repetitive, or poorly supported.

4. Sources
   - Check whether claims are supported by the provided sources.
   - Identify missing, irrelevant, or questionable sources.
   - Check whether the URLs listed actually come from the research.

5. Structure and Clarity
   - Check the Introduction, Key Findings, and Conclusion.
   - Identify confusing, repetitive, or poorly organized sections.

6. Overall Quality
   - Determine whether the report is factual, professional, and useful to the reader.

Return your critique using this structure:

## Overall Assessment
Briefly summarize the quality of the report.

## Strengths
List the strongest parts of the report.

## Problems Found
List specific problems and explain why they are problems.

## Missing Information
List important information that should be added.

## Source Issues
List any source or citation problems.

## Recommended Improvements
Give specific changes that should be made to improve the report.

Be factual and critical. Do not rewrite the entire report.
"""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()