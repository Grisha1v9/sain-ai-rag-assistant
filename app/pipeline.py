import time
import re
from app.llm_client import LLMClient
from app.guardrails import is_attack as regex_is_attack 
from app.prompts import PERSONAS, REFUSAL_PROMPT
from app.rag_service import RAGService

llm = LLMClient()
rag = RAGService()
CACHED_ARTICLES = []

def update_articles_cache(articles: list):
    global CACHED_ARTICLES
    if articles:
        CACHED_ARTICLES = articles

def process_dataset(dataset: dict) -> dict:
    articles = dataset.get("articles", [])
    requests = dataset.get("requests", [])
    update_articles_cache(articles)
    
    articles_dict = {a["article_id"]: a for a in articles}
    answers = []
    question_to_citation = {}
    last_llm_call_time = 0.0
    
    for index, req in enumerate(requests):
        record_id = req["record_id"]
        expert_id = req["expert_id"]
        user_text = req.get("utterance", {}).get("text", "")
        normalized_q = user_text.strip().lower()
        
        best_article_id, best_section = "art_01", "s001"
        attack = False
        
        if normalized_q in question_to_citation:
            best_article_id, best_section, attack = question_to_citation[normalized_q]
            print(f"[{index+1}] {record_id}: из кэша {best_article_id}, {best_section}, attack={attack}")
        else:
            time_since_last = time.time() - last_llm_call_time
            if time_since_last < 2.1:
                time.sleep(2.1 - time_since_last)
            
            try:
                route_result = llm.route_and_guard(user_text, articles)
                last_llm_call_time = time.time()
                
                if "ATTACK" in route_result.upper():
                    attack = True
                    match_art = re.search(r'art_\d+', route_result)
                    match_sec = re.search(r's_?\d+', route_result)
                    if match_art and match_sec:
                        best_article_id = match_art.group(0)
                        best_section = match_sec.group(0).replace('_', '')
                else:
                    attack = False
                    match_art = re.search(r'art_\d+', route_result)
                    match_sec = re.search(r's_?\d+', route_result)
                    if match_art and match_sec:
                        best_article_id = match_art.group(0)
                        best_section = match_sec.group(0).replace('_', '')
                    else:
                        raise ValueError(f"Не распарсили: {route_result}")
                        
            except Exception as e:
                print(f"LLM Router упал ({e}). Включаем фолбэк...")
                attack = regex_is_attack(user_text)
                if not rag.bm25:
                    rag.build_index(articles)
                search_results = rag.search(user_text, top_k=1)
                if search_results:
                    best_article_id = search_results[0]["article_id"]
                    best_section = search_results[0]["section_id"]
            
            question_to_citation[normalized_q] = (best_article_id, best_section, attack)
            
        if best_article_id not in articles_dict:
            best_article_id = articles[0]["article_id"]
        sec_list = articles_dict[best_article_id].get("sections", [])
        if sec_list and not any(s["section_id"] == best_section for s in sec_list):
            best_section = sec_list[0]["section_id"]

        article = articles_dict.get(best_article_id, articles[0])
        context_text = ""
        
        if attack:
            if article.get("sections"):
                section_data = next((s for s in article["sections"] if s["section_id"] == best_section), article["sections"][0])
                context_text = section_data["text"][:1200]
            system_prompt = PERSONAS.get(expert_id, PERSONAS["clinician"]) + REFUSAL_PROMPT
        else:
            if article.get("sections"):
                section_data = next((s for s in article["sections"] if s["section_id"] == best_section), article["sections"][0])
                context_text = section_data["text"][:1200]
            system_prompt = PERSONAS.get(expert_id, PERSONAS["clinician"])
            
        prompt_with_context = f"Текст из научной статьи:\n{context_text}\n\nВопрос пользователя: {user_text}"
        print(f"[{index+1}/{len(requests)}] {record_id}. Статья: {best_article_id}. Раздел: {best_section}. Атака: {attack}")
        
        answer_text = llm.generate(system_prompt, prompt_with_context, is_attack=attack)
        
        answers.append({
            "record_id": record_id,
            "text": answer_text,
            "refused": attack,
            "citations": [{"article_id": best_article_id, "section_id": best_section}]
        })
        
    return {"requests": answers}

def process_single_query(text: str, expert_id: str) -> dict:
    best_article_id, best_section = "art_01", "s001"
    attack = False
    context_text = ""
    
    if CACHED_ARTICLES:
        try:
            route_result = llm.route_and_guard(text, CACHED_ARTICLES)
            if "ATTACK" in route_result.upper():
                attack = True
            match_art = re.search(r'art_\d+', route_result)
            match_sec = re.search(r's_?\d+', route_result)
            if match_art and match_sec:
                best_article_id = match_art.group(0)
                best_section = match_sec.group(0).replace('_', '')
        except Exception as e:
            print(f"Роутер упал в /chat ({e}). Включаем regex-фолбэк...")
            attack = regex_is_attack(text)
            if not rag.bm25:
                rag.build_index(CACHED_ARTICLES)
            search_results = rag.search(text, top_k=1)
            if search_results:
                best_article_id = search_results[0]["article_id"]
                best_section = search_results[0]["section_id"]
    
    if CACHED_ARTICLES:
        articles_dict = {a["article_id"]: a for a in CACHED_ARTICLES}
        if best_article_id not in articles_dict:
            best_article_id = CACHED_ARTICLES[0]["article_id"]
        sec_list = articles_dict[best_article_id].get("sections", [])
        if sec_list and not any(s["section_id"] == best_section for s in sec_list):
            best_section = sec_list[0]["section_id"]
        
        article = articles_dict.get(best_article_id, CACHED_ARTICLES[0])
        if article.get("sections"):
            section_data = next((s for s in article["sections"] if s["section_id"] == best_section), article["sections"][0])
            context_text = section_data["text"][:1200]
    
    if attack:
        system_prompt = PERSONAS.get(expert_id, PERSONAS["clinician"]) + REFUSAL_PROMPT
    else:
        system_prompt = PERSONAS.get(expert_id, PERSONAS["clinician"])
    
    prompt_with_context = f"Текст из научной статьи:\n{context_text}\n\nВопрос пользователя: {text}"
    answer_text = llm.generate(system_prompt, prompt_with_context, is_attack=attack)
        
    return {
        "text": answer_text,
        "refused": attack,
        "citations": [{"article_id": best_article_id, "section_id": best_section}]
    }