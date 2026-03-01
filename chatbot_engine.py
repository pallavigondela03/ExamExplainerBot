import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

class ExamChatbot:
    def __init__(self):
        # 1. Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
        genai.configure(api_key=api_key)
        
        # 2. UPDATED MODEL NAMES (2026 Stable Versions)
        # We try Gemini 2.5 Flash first, as it is the current standard.
        try:
            # Using 'gemini-2.5-flash' which replaced 1.5
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini 2.5 Flash initialized.")
        except Exception:
            try:
                # Fallback to the latest version of Flash
                self.model = genai.GenerativeModel('gemini-flash-latest')
                print("✅ Using gemini-flash-latest.")
            except Exception:
                # Last resort fallback to Pro
                self.model = genai.GenerativeModel('gemini-2.5-pro')
                print("⚠️ Falling back to Gemini 2.5 Pro.")

        # 3. Professional System Instructions (The "Persona")
        self.system_instructions = (
            "You are an Academic Regulations Expert Bot. "
            "Your sole purpose is to explain examination patterns, evaluation rules, "
            "and grading policies based strictly on the provided context.\n\n"
            "STRICT RULES:\n"
            "1. If the answer is not in the context, say 'I do not have information on that policy.'\n"
            "2. Never predict a student's marks or pass/fail status.\n"
            "3. Never provide answers to exam questions or solve academic problems.\n"
            "4. Always mention the 'Source' filename in your explanation.\n"
            "5. Be concise, professional, and helpful."
        )

    def generate_response(self, user_query, retrieved_chunks):
        """
        Combines System Prompt + Context + User Query for a Grounded RAG Response.
        """
        # Format the retrieved chunks into a readable string for the AI
        if not retrieved_chunks:
            return "I'm sorry, I couldn't find any relevant context in the documents to answer that."

        context_text = "\n\n".join([
            f"--- SOURCE: {c['source']} ---\n{c['text']}" 
            for c in retrieved_chunks
        ])

        # Construct the final "Augmented" prompt
        final_prompt = f"""
        {self.system_instructions}

        RELEVANT ACADEMIC CONTEXT:
        {context_text}

        USER QUESTION: 
        {user_query}

        ASSISTANT'S GROUNDED RESPONSE:
        """

        try:
            # Generate response using the model
            response = self.model.generate_content(final_prompt)
            
            # Check if response has text (handles safety filters blocking the output)
            if response.text:
                return response.text
            else:
                return "I cannot provide an answer to that query due to safety constraints."
                
        except Exception as e:
            # Detailed error logging for the terminal
            print(f"❌ AI Generation Error: {str(e)}")
            return f"AI Error: I'm having trouble connecting to the brain. Error details: {str(e)}"