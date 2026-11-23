from app.graph import rag_graph, question_to_citation

CACHED_ARTICLES = []

def update_articles_cache(articles: list):
    global CACHED_ARTICLES
    if articles:
        CACHED_ARTICLES = articles

def process_single_query(text: str, expert_id: str) -> dict:
    if not CACHED_ARTICLES:
        return {"text": "База знаний пуста.", "refused": False, "citations": []}

    inputs = {
        "user_query": text,
        "expert_id": expert_id,
        "articles_meta": CACHED_ARTICLES,
        "articles_dict": {a["article_id"]: a for a in CACHED_ARTICLES},
        "cache_hit": False
    }
    
    result_state = rag_graph.invoke(inputs)
    
    return {
        "text": result_state["final_answer"],
        "refused": result_state["is_attack"],
        "citations": [{
            "article_id": result_state["article_id"], 
            "section_id": result_state["section_id"]
        }]
    }

def process_dataset(dataset: dict) -> dict:
    articles = dataset.get("articles", [])
    requests = dataset.get("requests", [])
    update_articles_cache(articles)
    
    answers = []
    for index, req in enumerate(requests):
        record_id = req["record_id"]
        expert_id = req["expert_id"]
        user_text = req.get("utterance", {}).get("text", "")
        
        res = process_single_query(user_text, expert_id)
        
        res["record_id"] = record_id
        answers.append(res)
        
    return {"requests": answers}