import json
import requests
from pydantic import BaseModel
from logging import getLogger
from src.common.utils import get_env_variable


logger = getLogger("OpenAIConnector")


class OpenAIConnector:

    def request_simple_text(
            self, 
            input_messages: list[dict[str, str]],
            max_tokens: int = 1000
    ) -> dict:
        data = {
            "messages": input_messages,
            "max_tokens": max_tokens,
        }
        return requests.post(
            url=get_env_variable("OPENAI_ENDPOINT"), 
            headers={
                "Content-Type": "application/json",
                "api-key": get_env_variable("OPENAI_KEY")
            }, 
            json=data
        )

    def request_wih_function_calling(
            self, 
            input_messages: list[dict[str, str]], 
            schema: dict,
            max_tokens: int = 1000
    ) -> dict:
        data = {
            "messages": input_messages,
            "max_tokens": max_tokens,
            "functions": [{
                "name": schema["title"],
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": schema["properties"],
                    "required": schema["required"],
                },
                "additionalProperties": False
            }],
            "function_call": {"name": schema["title"]}
        }
        return requests.post(
            url=get_env_variable("OPENAI_ENDPOINT"), 
            headers={
                "Content-Type": "application/json",
                "api-key": get_env_variable("OPENAI_KEY")
            }, 
            json=data
        )

    @staticmethod
    def create_human_message(prompt: str) -> dict[str, str]:
        return {
            "role": "user",
            "content": prompt
        }

    @staticmethod
    def create_image_input(prompt: str, image_url: str) -> dict[str, str]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt
                },
                {
                    "type": "input_image",
                    "image_url": image_url
                }
            ]
        }
    
