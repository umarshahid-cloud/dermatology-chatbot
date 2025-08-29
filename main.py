import os
import chainlit as cl
from dotenv import load_dotenv
from app.rag.rag_chain import RAGChain

load_dotenv()

# Optional debugpy
# if os.getenv("ENABLE_DEBUGPY", "0") == "1":
#     import debugpy
#     debugpy.listen(("localhost", 5678))
#     print("🔧 debugpy waiting on 5678...")

rag = RAGChain(temperature=0.0)

@cl.on_chat_start
async def start():
    await cl.Message("Dermatology bot ready. Ask MBBS-style questions.").send()

@cl.on_message
async def on_msg(msg: cl.Message):
    q = (msg.content or "").strip()

    # Start a streaming reply
    reply = cl.Message(content="")
    await reply.send()

    try:
        # 1) Embed
        question_vector = rag.embed_text(q)

        # 2)     Pinecone
        raw_documents = rag.search_index(question_vector, top_k=4)

        # 3) Normalize matches
        formatted_documents = rag.parse_results(raw_documents)

        # 4) Build context
        context = rag.format_context(formatted_documents)

        # 5) Build messages
        messages = rag.build_messages(q, context)

        # 6) Stream model output
        stream = rag.stream_answer(messages)
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                await reply.stream_token(content)

        await reply.update()

    except Exception:
        await reply.update(content="Sorry—something went wrong.")
        raise
