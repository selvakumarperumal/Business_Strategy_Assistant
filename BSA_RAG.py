import asyncio
from agno.agent import Agent, RunOutput
from agno.models.nvidia import Nvidia
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.google import GeminiEmbedder
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


# Load environment variables
load_dotenv("./.env")


class Settings(BaseSettings):
    NVIDIA_API_KEY: str
    NVIDIA_MODEL_NAME: str
    NVIDIA_EMBEDDING_MODEL_NAME: str
    GEMINI_API_KEY: str
    GEMINI_EMBEDDING_MODEL_NAME: str


settings = Settings()


# Gemini embedder needs `model=` not `id=`
gemini_embedder = GeminiEmbedder(
    api_key=settings.GEMINI_API_KEY,
    id=settings.GEMINI_EMBEDDING_MODEL_NAME,
)


# Chroma vector DB
vector_db = ChromaDb(
    embedder=gemini_embedder,
    collection="nvidia_embeddings_test",
    path="./chroma_nvidia_embedder",
    persistent_client=True
)


# Knowledge base
knowledge = Knowledge(
    vector_db=vector_db
)

# If you want, re-enable loading documents
knowledge.add_content(
    name="Recipes",
    url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
    metadata={"doc_type": "recipe_book"},
)


print("Vector DB loaded:", knowledge.vector_db)


# NVIDIA model
model = Nvidia(
    id=settings.NVIDIA_MODEL_NAME,
    api_key=settings.NVIDIA_API_KEY
)


# Agent (async)
agent = Agent(
    name="Business Strategy Assistant",
    model=model,
    markdown=True,
    add_knowledge_to_context=True,
    knowledge=knowledge,
    search_knowledge=True,
    description="An Expert assistant that answers query about business strategies using company documents.",
    instructions=[
        "You are business strategy expert assistant with deep knowledge of company's strategic documents.",
        "Always cite the sources from the documents when providing answers.",
        "If information is not available in the documents, politely inform the user that you don't have that information.",
        "Provide actionable recommendations based on the strategies outlined in the documents."

    ]
)


# Use an async wrapper!
async def main():
    run: RunOutput = await agent.arun(
        "What are some effective business strategies for market expansion?"
    )
    print(run.content)


# Run async
asyncio.run(main())