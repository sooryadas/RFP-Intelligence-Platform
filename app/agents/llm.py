"""Shared language model used by the RFP agent nodes."""

from langchain_groq import ChatGroq

from app.config import Settings


llm = ChatGroq(
    model=Settings.GROQ_MODEL,
    api_key=Settings.GROQ_API_KEY,
)
