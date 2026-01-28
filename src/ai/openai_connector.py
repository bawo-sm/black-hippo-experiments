import json
from openai import OpenAI
from pydantic import BaseModel
from src.common.utils import get_env_variable


class OpenAIConnector:

    def __init__(self):
        self.__client = OpenAI(api_key=get_env_variable("OPENAI_KEY"))
    
    def request_wih_function_calling(
            self, 
            input_messages: list[dict[str, str]], 
            schema: BaseModel,
            llm: str
    ) -> dict:
        schema = schema.model_json_schema()
        response = self.__client.responses.create(
            model=llm,
            input=input_messages,
            max_tool_calls=1,
            tools=[{
                "name": schema["title"],
                "type": "function",
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": schema["properties"],
                    "required": schema["required"],
                },
                "additionalProperties": False
            }],
            tool_choice={"type": "function", "name": schema["title"]}
        )
        return json.loads(response.output[0].arguments)

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