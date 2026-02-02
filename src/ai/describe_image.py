import json
from pydantic import BaseModel, Field
from src.ai.openai_connector import OpenAIConnector
from src.ai.prompts.prompts_manager import prompts_manager


class AnswerSchema(BaseModel):
    """Answer shcema for image description"""
    look: str = Field(description="What does it look like?")
    potential_usage: str = Field(description="How people can use this item?")
    materials: str = Field(description="What is the item made from?")


def describe_image(image_url: str) -> AnswerSchema:
    prompt = prompts_manager.describe_image.format(answer_schema=AnswerSchema.model_json_schema())
    messages = [
        OpenAIConnector.create_image_input(
            prompt=prompt,
            image_url=image_url
        )
    ]
    raw_answer = OpenAIConnector().request_wih_function_calling(
        input_messages=messages,
        schema=AnswerSchema,
    )
    raw_answer = json.loads(raw_answer.json()["choices"][0]["message"]["function_call"]["arguments"])
    return AnswerSchema(**raw_answer)
