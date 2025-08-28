import os
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

from app.rag.rag_chain import RAGChain 

import os
if os.getenv("ENABLE_DEBUGPY", "0") == "1":
    import debugpy
    debugpy.listen(("localhost", 5678))
    print("🔧 debugpy waiting on 5678...")

rag = RAGChain(
    index_name=os.getenv("PINECONE_INDEX", "derma-chatbot"),
    namespace="dermatology",
    embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
    chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
)

@cl.on_chat_start
async def start():
    await cl.Message(
        "Dermatology bot ready. Ask MBBS-style questions."
    ).send()

@cl.on_message
async def on_msg(msg: cl.Message):
    q = (msg.content or "").strip()
    reply = cl.Message(content="")
    await reply.send()
    try:
        for tok in rag.stream_answer(q, k=4):
            await reply.stream_token(tok)
        await reply.update()
    except Exception:
        await reply.update(content="Sorry—something went wrong.")
        raise
