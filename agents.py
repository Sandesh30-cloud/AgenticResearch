import os

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import web_search, scrape_url, customer_search_tool
from dotenv import load_dotenv
from customer_config import CustomerConfig

load_dotenv()

# gemini-2.0-flash free-tier quota is often 0; 2.5-flash works on the same API key
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    max_retries=3,
    timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
)


def llm_for_customer(config: CustomerConfig):
    """Create the customer's model client without sharing mutable request state."""
    if (config.model == GEMINI_MODEL and config.llm_timeout_seconds == float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
            and config.retry_attempts == 3):
        return llm
    return ChatGoogleGenerativeAI(
        model=config.model,
        temperature=0,
        max_retries=config.retry_attempts,
        timeout=config.llm_timeout_seconds,
    )


#1st agent 
def build_search_agent(config: CustomerConfig | None = None):
    search_tool = (customer_search_tool(config.customer_id, max_results=config.max_sources,
                                        retry_attempts=config.retry_attempts,
                                        calls_per_minute=config.search_rate_limit_per_minute)
                   if config else web_search)
    return create_agent(
        model=llm_for_customer(config) if config else llm,
        tools=[search_tool]
    )

#2nd agent 

def build_reader_agent(config: CustomerConfig | None = None):
    return create_agent(
        model=llm_for_customer(config) if config else llm,
        tools = [scrape_url]
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


def build_writer_chain(config: CustomerConfig, research_questions: list[str] | None = None,
                      deliverables: list[str] | None = None,
                      customer_context: str | None = None):
    questions = "\n".join(f"- {question}" for question in (research_questions or [])) or "Not specified; derive sensible questions from the topic."
    sections = "\n".join(f"- {section}" for section in (deliverables or [])) or "- Introduction\n- Key findings\n- Conclusion\n- Sources"
    context = customer_context or config.research_context() or "No additional customer context provided."
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.system_prompt),
        ("human", f"""Prepare a detailed, factual research report for this customer.

Customer: {config.display_name}
Customer context:
{context}

Research topic: {{topic}}

Research questions:
{questions}

Research gathered:
{{research}}

Required report sections:
{sections}

Use the required sections in order. Address every research question, separate verified facts from analysis, identify evidence gaps, and include source URLs. Do not invent dates, prices, or claims when sources do not provide them."""),
    ])
    return prompt | llm_for_customer(config) | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()


def build_critic_chain(config: CustomerConfig):
    return critic_prompt | llm_for_customer(config) | StrOutputParser()

