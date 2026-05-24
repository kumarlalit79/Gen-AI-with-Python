from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# model
model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# JSON Schema
review_schema = {
    "title": "Review",
    "description": "Structured review analysis",
    "type": "object",
    "properties": {

        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Key themes discussed in the review"
        },

        "summary": {
            "type": "string",
            "description": "Brief summary of the review"
        },

        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Sentiment of the review"
        },

        "pros": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Advantages mentioned in the review"
        },

        "cons": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Disadvantages mentioned in the review"
        },

        "name": {
            "type": "string",
            "description": "Reviewer name if available"
        }
    },

    "required": ["key_themes", "summary", "sentiment"]
}

# structured model
structured_model = model.with_structured_output(review_schema)

# review text
review = """
The laptop performance is excellent and battery life is impressive.
The display quality is also very sharp.

However, the laptop becomes hot sometimes and is a little expensive.

- Aman Verma
"""

# invoke
result = structured_model.invoke(review)

print(result)