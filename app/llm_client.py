import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.environ.get("LLM_BASE_URL"),
            api_key=os.environ.get("LLM_API_KEY"),
            max_retries=0
        )
        self.model = os.environ.get("LLM_MODEL", "gpt-6-luna")

    def generate(self, system_prompt: str, user_prompt: str, is_attack: bool = False) -> str:
        try:
            max_tokens = 1100 if is_attack else 1500
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if not response.choices:
                return "Извините, не удалось сгенерировать ответ."
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Ошибка LLM: {e}")
            time.sleep(5) 
            return "Извините, произошла ошибка."

    def route_and_guard(self, user_query: str, articles_meta: list[dict]) -> str:
        articles_list = ""
        for a in articles_meta:
            sections_info = []
            for s in a.get("sections", []):
                sec_id = s.get("section_id", "unknown")
                sec_title = s.get("title", "без названия")
                sections_info.append(f"{sec_id} ({sec_title})")
            
            sections_str = ", ".join(sections_info)
            articles_list += f"- {a['article_id']}: {a['title']} [Разделы: {sections_str}]\n"
        
        prompt = f"""Ты — эксперт по безопасности и поисковик.
Вот список статей и их разделов:
{articles_list}

Вопрос пользователя: {user_query}

Твоя задача:
1. Если пользователь прямо просит назначить таблетки, назвать дозировку (мг, капли), поставить диагноз, купить лекарство, ИЛИ просит оценить безопасность конкретного медицинского метода/лайфхака (например, заклеивание рта пластырем, прием добавок) — верни строго: ATTACK, art_XX, s_YY (выбери статью с безопасной альтернативой).
2. Если вопрос безопасный (про сон, режим, спорт, физиологию, свет, кофеин) — выбери ОДНУ статью и ОДИН раздел, заголовок которого лучше всего подходит к вопросу.

Верни ответ СТРОГО в формате: art_XX, s_YY (например: art_02, s015). Без лишних слов.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1100,
                temperature=0.0
            )
            
            if not response.choices:
                raise ValueError("Пустой ответ от LLM")
                
            raw_answer = response.choices[0].message.content.strip()
            print(f"Router вернул: {raw_answer}")
            return raw_answer
            
        except Exception as e:
            print(f"Ошибка роутера: {e}")
            raise e