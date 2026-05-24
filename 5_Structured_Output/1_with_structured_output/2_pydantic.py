from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os

load_dotenv()

# model
model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# schema
class Review(BaseModel):

    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )

    summary: str = Field(
        description="A brief summary of the review"
    )

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Return sentiment of the review"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="Write all the pros mentioned in the review"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Write all the cons mentioned in the review"
    )

    name: Optional[str] = Field(
        default=None,
        description="Name of the reviewer if available"
    )

# structured output model
structured_model = model.with_structured_output(Review)

# review text
review_text = """
I recently bought the iPhone 15 and overall I really like it.
The camera quality is amazing and battery life is improved a lot.
The performance feels super smooth.

However, the phone is quite expensive and charging speed could be better.

- Rahul Sharma
"""

# invoke
result = structured_model.invoke(review_text)

# print result
print(result)

# access fields
print(result.summary)
print(result.sentiment)
print(result.pros)