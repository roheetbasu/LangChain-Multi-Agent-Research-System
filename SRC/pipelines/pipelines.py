from SRC.agents.agents import build_read_agent,build_search_agent,writer_chain,critic_chain

def run_research_pipelines(topic: str):
    state = {}
    
    # search agent working
    print("\n"+"="*50)
    print("Step 1: Search Agent is working ")
    print("="*50)
    
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [('user', f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content
    
    print("\n search results ",state['search_results'])
    
    # step 2 reader agent
    print("\n"+"="*50)
    print("Step 2: Read Agent is scraping top resources ")
    print("="*50)
    
    reader_agent = build_read_agent()
    reader_result= reader_agent.invoke({
        "messages" :   [('user', 
            f"Based on following search results about '{topic}',"
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"    
            )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content
    print("\nscraped content:  \n", state['scraped_content'])
    
    # step 3 - writer chain
    print("\n"+"="*50)
    print("Step 3: Writer is drafting report ... ")
    print("="*50)

    research_combined = (
        f"SEARCHED RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : {state['scraped_content']}"
    )    
    
    state['report'] = writer_chain.invoke({
        'topic' : topic,
        'research' : research_combined
    })
    
    print("\nFinal Report\n",state['report'])
    
    # critic report
    print("\n"+"="*50)
    print("Step 4: Critic is reviewing report ... ")
    print("="*50)

    state['feedback'] = critic_chain.invoke({
        'research':research_combined,
        'report':state['report'],
        'topic':topic
    })
    
    print("\n critic report \n", state['feedback'])
    
    return state
