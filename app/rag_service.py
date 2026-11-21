from rank_bm25 import BM25Okapi
import re

class RAGService:
    def __init__(self):
        self.bm25 = None
        self.corpus_meta = []

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^а-яёa-z\s]', '', text)
        
        words = text.split()
        stems = []
        for w in words:
            if len(w) > 4:
                stems.append(w[:-2])
            else:
                stems.append(w)
        return stems

    def build_index(self, articles: list[dict]):
        corpus_texts = []
        self.corpus_meta = []

        for article in articles:
            article_title = article.get("title", "")
            
            for section in article.get("sections", []):
                text = section.get("text", "")
                section_title = section.get("title", "")
                full_text = f"{article_title} {section_title} {text}"
                
                corpus_texts.append(self._tokenize(full_text))
                self.corpus_meta.append({
                    "article_id": article.get("article_id"),
                    "section_id": section.get("section_id"),
                    "raw_text": text
                })

        if corpus_texts:
            self.bm25 = BM25Okapi(corpus_texts)
    
    def search(self, query: str, top_k: int = 1) -> list[dict]:
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append(self.corpus_meta[idx])
        return results