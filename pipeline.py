from agents import build_reader_agent, build_search_agent, build_writer_chain, build_critic_chain
from customer_config import load_customer_config
from resilience import run_stage
from schemas import ResearchRequest, ResearchResponse
from datetime import datetime, timezone

def run_research_pipeline(topic: str, customer_id: str = "default",
                          research_questions: list[str] | None = None,
                          deliverables: list[str] | None = None,
                          customer_context: str = "") -> dict:

    request = ResearchRequest.model_validate({
        "topic": topic.strip(),
        "customer_id": customer_id,
        "customer_context": customer_context,
        "research_questions": research_questions or [],
        "deliverables": deliverables or [],
    })
    topic = request.topic
    config = load_customer_config(request.customer_id)

    state = {}

    #search agent working
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    try:
        def search_operation():
            agent = build_search_agent(config)
            return agent.invoke({"messages": [("user",
                f"Find recent, reliable and detailed information about: {topic}\n"
                f"Customer context:\n{request.customer_context or config.research_context()}\n"
                f"Research questions:\n{chr(10).join(request.research_questions)}")]})
        search_result = run_stage("search", search_operation, executions=state.setdefault("executions", {}), max_attempts=config.retry_attempts)
        state["search_results"] = search_result['messages'][-1].content
    except Exception as exc:
        state["search_results"] = f"Search unavailable: {exc}. No verified search sources were returned."

    print("\n search result ",state['search_results'])

    #step 2 - reader agent
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    try:
        def reader_operation():
            agent = build_reader_agent(config)
            return agent.invoke({"messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL related to the customer's objective and research questions, then scrape it for deeper content.\n"
                f"Customer context:\n{request.customer_context or config.research_context()}\n"
                f"Research questions:\n{chr(10).join(request.research_questions)}\n\n"
                f"Search Results:\n{state['search_results'][:800]}")]})
        reader_result = run_stage("reader", reader_operation, executions=state["executions"], max_attempts=config.retry_attempts)
        state['scraped_content'] = reader_result['messages'][-1].content
    except Exception as exc:
        state['scraped_content'] = f"Reader unavailable: {exc}. Use the search snippets below and disclose that source pages could not be accessed."

    print("\nscraped content: \n", state['scraped_content'])

    #step 3 - writer chain

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    try:
        writer = build_writer_chain(config, request.research_questions, request.deliverables,
                                    request.customer_context or config.research_context())
        state["report"] = run_stage("writer", lambda: writer.invoke({
        "topic" : topic,
        "research" : research_combined
        }), executions=state["executions"], max_attempts=config.retry_attempts)
    except Exception as exc:
        state["report"] = f"Report generation failed ({type(exc).__name__}): {exc}"

    print("\n Final Report\n",state['report'])

    #critic report

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    try:
        state["feedback"] = run_stage("critic", lambda: build_critic_chain(config).invoke({
        "report":state['report']
        }), executions=state["executions"], max_attempts=config.retry_attempts)
    except Exception as exc:
        state["feedback"] = f"Critic review unavailable ({type(exc).__name__}): {exc}"

    print("\n critic report \n", state['feedback'])

    response = ResearchResponse(
        topic=topic,
        customer_id=config.customer_id,
        customer_context=request.customer_context or config.research_context(),
        research_questions=request.research_questions,
        deliverables=request.deliverables,
        search_results=state.get("search_results", ""),
        scraped_content=state.get("scraped_content", ""),
        report=state.get("report", ""),
        feedback=state.get("feedback", ""),
        executions=state.get("executions", {}),
        created_at=datetime.now(timezone.utc),
    )
    return response.model_dump(mode="json")



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)

