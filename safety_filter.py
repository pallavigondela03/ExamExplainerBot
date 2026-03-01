import re

class SafetyFilter:
    BLACK_LIST_KEYWORDS = [
        "give me the answer to question", "predict my grade", 
        "what is the answer for", "solve this exam", "leak"
    ]

    @staticmethod
    def is_safe(query: str) -> bool:
        query_lower = query.lower()
        # Prevent exam answer seeking
        if any(keyword in query_lower for keyword in SafetyFilter.BLACK_LIST_KEYWORDS):
            return False
        # Prevent requests for personal grade predictions
        if re.search(r"(will i pass|my grade|predict|score)", query_lower):
            if not re.search(r"(rule|policy|criteria)", query_lower):
                return False
        return True