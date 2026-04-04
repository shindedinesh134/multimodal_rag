"""
Image processing and VLM integration for generating text summaries
"""

import os
import base64
import io
from typing import Dict, Any, Optional, List
from PIL import Image
import requests
from loguru import logger


class ImageProcessor:
    """Process images using Vision Language Model (VLM) to generate text summaries"""
    
    def __init__(self, provider: str = "replicate", model: str = "llava-13b"):
        """
        Initialize VLM processor.
        
        Args:
            provider: 'replicate' or 'openai'
            model: Model name/identifier
        """
        self.provider = provider
        self.model = model
        
        # Check API keys
        if provider == "replicate":
            self.api_token = os.getenv("REPLICATE_API_TOKEN")
            if not self.api_token:
                logger.warning("REPLICATE_API_TOKEN not set. Image summaries will be basic.")
        elif provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not set. Image summaries will be basic.")
    
    def process_image(self, image: Image.Image, context: Optional[str] = None) -> str:
        """
        Generate text summary for an image using VLM.
        
        Args:
            image: PIL Image object
            context: Optional context about where image came from (e.g., "cooling system diagram")
        
        Returns:
            Text description of the image
        """
        try:
            if self.provider == "replicate":
                return self._process_with_replicate(image, context)
            elif self.provider == "openai":
                return self._process_with_openai(image, context)
            else:
                return self._generate_basic_description(image, context)
        except Exception as e:
            logger.error(f"VLM processing failed: {e}")
            return self._generate_basic_description(image, context)
    
    def _process_with_replicate(self, image: Image.Image, context: Optional[str]) -> str:
        """Use Replicate's LLaVA model"""
        import replicate
        
        # Convert PIL to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f"""You are analyzing a technical diagram or image from an engine cooling system manual.
Please describe what you see in detail, focusing on:
- What type of diagram/image this is (schematic, flow diagram, component photo, chart, etc.)
- Key components or elements shown
- Any labels, numbers, or annotations
- The purpose or function this image illustrates

Context: {context if context else "From engine cooling technical manual"}

Image description:"""
        
        # Run LLaVA on Replicate
        output = replicate.run(
            self.model,
            input={
                "image": f"data:image/png;base64,{img_base64}",
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 300
            }
        )
        
        description = "".join(output) if isinstance(output, list) else str(output)
        return description.strip()
    
    def _process_with_openai(self, image: Image.Image, context: Optional[str]) -> str:
        """Use OpenAI GPT-4V"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        # Convert PIL to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        prompt = f"""You are analyzing a technical diagram or image from an engine cooling system manual.
Please describe what you see in detail, focusing on:
- What type of diagram/image this is
- Key components or elements shown
- Any labels, numbers, or annotations
- The purpose or function this image illustrates

Context: {context if context else "From engine cooling technical manual"}

Provide a concise but thorough description:"""
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_basic_description(self, image: Image.Image, context: Optional[str]) -> str:
        """Fallback: generate basic description without VLM"""
        width, height = image.size
        mode = image.mode
        
        description = f"[Image - Size: {width}x{height}, Mode: {mode}"
        if context:
            description += f", Context: {context}"
        description += "]"
        
        return description
    
    def process_document_images(self, parsed_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all images in a parsed document and add summaries.
        
        Returns:
            Updated parsed_doc with 'summary' field for each image
        """
        images = parsed_doc.get("images", [])
        logger.info(f"Processing {len(images)} images with VLM...")
        
        for idx, image_data in enumerate(images):
            context = f"Page {image_data.get('page', 'unknown')}, {parsed_doc.get('filename', 'document')}"
            
            if "image" in image_data:
                summary = self.process_image(image_data["image"], context)
                image_data["summary"] = summary
                logger.debug(f"Image {idx+1}/{len(images)} processed")
            else:
                image_data["summary"] = "[Image data not available]"
        
        return parsed_doc