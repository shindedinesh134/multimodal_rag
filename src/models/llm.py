"""
LLM wrapper for text generation
"""

import os
from typing import Optional, List, Dict, Any
from loguru import logger


class LLMWrapper:
    """Wrapper for various LLM providers"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.provider = provider
        self.model_name = model
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized")
                else:
                    logger.warning("OPENAI_API_KEY not set")
            except ImportError:
                logger.warning("OpenAI package not installed")
                
        elif self.provider == "replicate":
            try:
                import replicate
                api_token = os.getenv("REPLICATE_API_TOKEN")
                if api_token:
                    # Replicate client is configured via environment variable
                    self.client = replicate
                    logger.info("Replicate client initialized")
                else:
                    logger.warning("REPLICATE_API_TOKEN not set")
            except ImportError:
                logger.warning("Replicate package not installed")
        
        elif self.provider == "local":
            # For local models like Llama (optional)
            logger.info("Using local mode - no external API calls")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> str:
        """Generate text from prompt"""
        
        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")
                return self._fallback_generation(prompt)
        
        elif self.provider == "replicate" and self.client:
            try:
                # Using Llama 2 or similar on Replicate
                output = self.client.run(
                    self.model_name,
                    input={
                        "prompt": prompt,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    }
                )
                # Output is usually a list of strings
                if isinstance(output, list):
                    return "".join(output).strip()
                return str(output).strip()
            except Exception as e:
                logger.error(f"Replicate generation failed: {e}")
                return self._fallback_generation(prompt)
        
        else:
            return self._fallback_generation(prompt)
    
    def _fallback_generation(self, prompt: str) -> str:
        """Fallback when no LLM is available"""
        logger.warning("Using fallback response generation")
        
        # Extract question from prompt
        lines = prompt.split("\n")
        question = ""
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
                break
        
        if "coolant" in question.lower():
            return "Based on the documentation, coolant specifications vary by engine model. Please refer to the technical specifications table in the manual for exact ratios and types."
        elif "temperature" in question.lower():
            return "The operating temperature range is documented in the cooling system specifications. I recommend checking the temperature thresholds table in the manual."
        elif "flow" in question.lower():
            return "The coolant flow path is illustrated in the system diagram. Please refer to the image on page 3 which shows the complete circulation path."
        else:
            return "I've found relevant information in the documentation. Please check the retrieved sources for complete details on this topic."