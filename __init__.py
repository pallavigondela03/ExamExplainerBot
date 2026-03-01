import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class ExamChatbot:
    def __init__(self):
        # 1. Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
        # 2. FORCE RE-CONFIGURATION
        genai.configure(api_key=api_key)
        
        # 3. USE THE BASE MODEL NAME (Most compatible)
        # If 'gemini-1.5-flash' fails, we fall back to 'gemini-pro'
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            # Test if model is reachable
            print("✅ Gemini 1.5 Flash initialized.")
        except Exception:
            self.model = genai.GenerativeModel('gemini-pro')
            print("⚠️ Falling back to Gemini Pro.")
        
        # Professional System Instructions
        self.system_instructions = (
            "You are an Academic Regulations Expert Bot. "
            "Your sole purpose is to explain examination patterns, evaluation rules, "
            "and grading policies based strictly on the provided context.\n\n"
            "STRICT RULES:\n"
            "1. If the answer is not in the context, say 'I do not have information on that policy.'\n"
            "2. Never predict a student's marks or pass/fail status.\n"
            "3. Never provide answers to exam questions or solve academic problems.\n"
            "4. Always mention the 'Source' filename in your explanation."
        )

    def generate_response(self, user_query, retrieved_chunks):
        context_text = "\n\n".join([
            f"--- SOURCE: {c['source']} ---\n{c['text']}" 
            for c in retrieved_chunks
        ])

        final_prompt = f"""
        {self.system_instructions}

        RELEVANT ACADEMIC CONTEXT:
        {context_text}

        USER QUESTION: 
        {user_query}

        ASSISTANT'S GROUNDED RESPONSE:
        """

        try:
            # We add a safety check here
            response = self.model.generate_content(final_prompt)
            return response.text
        except Exception as e:
            # This helps us see if it's a 404 or a Quota issue
            return f"AI Error: {str(e)}"