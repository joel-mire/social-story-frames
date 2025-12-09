import json
from typing import Union, List

class OutputParser:
    """Utilities for parsing LLM generation outputs"""
    
    @staticmethod
    def clean_json_markdown(text: str) -> str:
        """Remove markdown code fences from JSON output"""
        return text.lstrip("```json").rstrip("```").strip()
    
    @staticmethod
    def parse_single_response(output: Union[str, dict]) -> str:
        """
        Parse a single-response LLM output.
        Used by SingleOutputTaskManager and ImplausibleSingleOutputTaskManager.
        
        Args:
            output: Raw LLM output (string or dict with 'output' key)
        
        Returns:
            The parsed response text
        """
        if isinstance(output, dict):
            output = output.get("output", output)
        
        cleaned = OutputParser.clean_json_markdown(output)
        parsed = json.loads(cleaned)
        
        return parsed['response']
    
    @staticmethod
    def parse_multi_response(output: Union[str, dict]) -> List[str]:
        """
        Parse a multi-response LLM output.
        Used by BatchMultiOutputTaskManager.
        
        Args:
            output: Raw LLM output (string or dict)
        
        Returns:
            List of response texts
        """
        cleaned = OutputParser.clean_json_markdown(output)
        parsed = json.loads(cleaned)
        
        responses = parsed['responses']
        if isinstance(responses, list):
            return responses
        else:
            return [responses]