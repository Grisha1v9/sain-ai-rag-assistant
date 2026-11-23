import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from app.llm_client import LLMClient
from app.guardrails import is_attack as regex_is_attack
from app.rag_service import RAGService
from app.prompts import PERSONAS, REFUSAL_PROMPT

llm = LLMClient()
rag = RAGService()

question_to_citation = {}


class AgentState(TypedDict):
    user_query: str
    expert_id: str
    articles_meta: List[Dict[str, Any]]
    articles_dict: Dict[str, Any]
    
    cache_hit: bool
    
    is_attack: bool
    article_id: str
    section_id: str
    
    context_text: str
    system_prompt: str
    user_prompt: str
    
    final_answer: str


def check_cache_node(state: AgentState) -> Dict[str, Any]:
    """Узел 1: Проверяем, отвечали ли мы уже на этот вопрос (Compliance)."""
    normalized_q = state["user_query"].strip().lower()
    if normalized_q in question_to_citation:
        art, sec, att = question_to_citation[normalized_q]
        return {
            "cache_hit": True,
            "article_id": art,
            "section_id": sec,
            "is_attack": att
        }
    return {"cache_hit": False}

def router_and_guard_node(state: AgentState) -> Dict[str, Any]:
    """Узел 2: LLM-роутер + Regex фолбэк + BM25 фолбэк."""
    user_text = state["user_query"]
    articles = state["articles_meta"]
    
    attack = False
    best_article_id, best_section = "art_01", "s001"
    
    try:
        route_result = llm.route_and_guard(user_text, articles)
        if "ATTACK" in route_result.upper():
            attack = True
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
            
    return {
        "is_attack": attack,
        "article_id": best_article_id,
        "section_id": best_section
    }

def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    """Узел 3: Валидация ID и извлечение текста статьи."""
    articles_dict = state["articles_dict"]
    best_article_id = state["article_id"]
    best_section = state["section_id"]
    
    if best_article_id not in articles_dict:
        best_article_id = list(articles_dict.keys())[0] if articles_dict else "art_01"
        
    sec_list = articles_dict.get(best_article_id, {}).get("sections", [])
    if sec_list and not any(s["section_id"] == best_section for s in sec_list):
        best_section = sec_list[0]["section_id"]
        
    article = articles_dict.get(best_article_id, {})
    context_text = ""
    if article.get("sections"):
        section_data = next((s for s in article["sections"] if s["section_id"] == best_section), article["sections"][0])
        context_text = section_data["text"][:1200]
        
    return {
        "article_id": best_article_id,
        "section_id": best_section,
        "context_text": context_text
    }

def build_prompt_node(state: AgentState) -> Dict[str, Any]:
    """Узел 4: Сборка системного и пользовательского промпта."""
    expert_id = state["expert_id"]
    is_attack = state["is_attack"]
    context_text = state["context_text"]
    user_text = state["user_query"]
    
    base_prompt = PERSONAS.get(expert_id, PERSONAS["clinician"])
    system_prompt = base_prompt + REFUSAL_PROMPT if is_attack else base_prompt
    user_prompt = f"Текст из научной статьи:\n{context_text}\n\nВопрос пользователя: {user_text}"
    
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }

def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """Узел 5: Финальная генерация ответа."""
    answer_text = llm.generate(
        state["system_prompt"], 
        state["user_prompt"], 
        is_attack=state["is_attack"]
    )
    return {"final_answer": answer_text}

def save_to_cache_node(state: AgentState) -> Dict[str, Any]:
    """Узел 6: Сохраняем результат в кэш для следующих одинаковых запросов."""
    if not state.get("cache_hit"):
        normalized_q = state["user_query"].strip().lower()
        question_to_citation[normalized_q] = (
            state["article_id"], 
            state["section_id"], 
            state["is_attack"]
        )
    return {}


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("router_and_guard", router_and_guard_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("save_to_cache", save_to_cache_node)

    workflow.set_entry_point("check_cache")

    workflow.add_conditional_edges(
        "check_cache",
        lambda state: "retrieve_context" if state.get("cache_hit") else "router_and_guard",
        {
            "retrieve_context": "retrieve_context",
            "router_and_guard": "router_and_guard"
        }
    )

    workflow.add_edge("router_and_guard", "retrieve_context")
    workflow.add_edge("retrieve_context", "build_prompt")
    workflow.add_edge("build_prompt", "generate_answer")
    workflow.add_edge("generate_answer", "save_to_cache")
    workflow.add_edge("save_to_cache", END)

    return workflow.compile()


rag_graph = build_graph()