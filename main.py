import os
from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from app.rag.rag_chain import RAGChain

MAX_TURNS = 8

def get_history():
    return cl.user_session.get("history") or []

def set_history(hist):
    cl.user_session.set("history", hist)

def add_turn(role, content):
    hist = get_history()
    hist.append({"role": role, "content": content})
    if len(hist) > MAX_TURNS * 2:
        hist[:] = hist[-MAX_TURNS*2:]
    set_history(hist)

rag = RAGChain(temperature=0.0)


@cl.on_chat_start
async def start():
    cl.user_session.set("history", []) 
    await cl.Message(
        content="Hello! I’m your dermatology study assistant. How can I help you today?"
    ).send()


@cl.on_message
async def on_msg(msg: cl.Message):
    q = (msg.content or "").strip()

    reply = cl.Message(content="")
    await reply.send()

    try:
        # 1) Embed question
        q_vec = rag.embed_text(q)

        # 2) Search Pinecone
        raw = rag.search_index(q_vec, top_k=10)

        # 3) Build (system + latest user-with-context)
        base_messages = rag.build_messages_from_result(q, raw)

        # 4) Inject prior turns
        messages = [base_messages[0], *get_history(), base_messages[1]]

        # 5) Stream model output
        stream = rag.stream_answer(messages)
        collected = []
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                collected.append(content)
                await reply.stream_token(content)

        final_answer = "".join(collected)
        await reply.update()

        # 6) Save this turn to history
        add_turn("user", q)
        add_turn("assistant", final_answer)

    except Exception:
        await reply.update(content="Sorry—something went wrong.")
        raise
