import asyncio
import hashlib
import logging
import queue
import time
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_backend import (
    MAX_PDF_BYTES,
    capability_status,
    chatbot,
    delete_thread_data,
    get_document_metadata,
    ingest_pdf,
    reserve_chat_request,
    submit_async_task,
    touch_thread,
)

MAX_INPUT_LENGTH = 2_000
MAX_REQUESTS_PER_SESSION = 30
MAX_TURN_SECONDS = 120

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Chatbot")
st.caption("Live search, stock lookup, PDF RAG, memory, and safe local MCP tools")


def new_chat() -> None:
    previous_thread = st.session_state.thread_id
    delete_thread_data(previous_thread)
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.message_history = []
    st.session_state.document_digest = None
    st.session_state.document_summary = None
    st.session_state.uploader_key += 1


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content) if content else ""


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "document_digest" not in st.session_state:
    st.session_state.document_digest = None
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

touch_thread(st.session_state.thread_id)
status = capability_status()

with st.sidebar:
    st.header("Capabilities")
    st.markdown(
        "- Live news and web search\n"
        "- Calculator\n"
        "- PDF question answering\n"
        "- Conversation memory\n"
        "- Local read-only MCP utilities"
    )
    if status["stock_api"]:
        st.success("Structured stock quotes enabled")
    else:
        st.info("Stock questions use web search until Alpha Vantage is configured.")
    st.caption(f"MCP: {status['mcp']}")

    if st.button("New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.subheader("Ask about a PDF")
    st.caption(
        "Maximum 5 MB and 30 pages. PDF text is sent to Gemini for embeddings and answers. "
        "Do not upload confidential or sensitive documents."
    )
    uploaded_pdf = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"],
        key=f"pdf-uploader-{st.session_state.uploader_key}",
    )

    if uploaded_pdf is not None:
        if uploaded_pdf.size > MAX_PDF_BYTES:
            st.error("This PDF exceeds the 5 MB limit.")
        else:
            file_bytes = uploaded_pdf.getvalue()
            digest = hashlib.sha256(file_bytes).hexdigest()
            if digest == st.session_state.document_digest:
                st.info("This PDF is already indexed for the current chat.")
            elif st.button("Index PDF", use_container_width=True):
                try:
                    touch_thread(st.session_state.thread_id)
                    with st.spinner("Reading and indexing PDF..."):
                        summary = ingest_pdf(
                            file_bytes,
                            thread_id=st.session_state.thread_id,
                            filename=uploaded_pdf.name,
                        )
                    st.session_state.document_digest = digest
                    st.session_state.document_summary = summary
                    st.success(
                        f"Indexed {summary['filename']}: {summary['pages']} pages, "
                        f"{summary['chunks']} chunks."
                    )
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    logger.exception("PDF indexing failed")
                    st.error("The PDF could not be indexed. Check the owner logs for details.")

    document_summary = get_document_metadata(st.session_state.thread_id)
    if document_summary:
        st.success(
            f"Active document: {document_summary['filename']} "
            f"({document_summary['pages']} pages)"
        )

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question")

if user_input:
    if len(user_input) > MAX_INPUT_LENGTH:
        st.error(f"Please limit your message to {MAX_INPUT_LENGTH:,} characters.")
        st.stop()

    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.warning("This demo has reached its message limit for this browser session.")
        st.stop()

    if not reserve_chat_request():
        st.warning("This public demo has reached its daily request limit.")
        st.stop()

    st.session_state.request_count += 1
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    thread_id = st.session_state.thread_id
    touch_thread(thread_id)
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
        "recursion_limit": 8,
    }

    try:
        with st.chat_message("assistant"):
            tool_status = {"box": None}

            def response_stream():
                events: queue.Queue = queue.Queue(maxsize=100)

                async def emit(event) -> None:
                    await asyncio.to_thread(events.put, event, True, 1)

                async def run_graph():
                    try:
                        async with asyncio.timeout(MAX_TURN_SECONDS - 10):
                            async for message_chunk, metadata in chatbot.astream(
                                {
                                    "messages": [HumanMessage(content=user_input)],
                                    "thread_id": thread_id,
                                },
                                config=config,
                                stream_mode="messages",
                            ):
                                await emit((message_chunk, metadata))
                    except Exception as exc:
                        try:
                            await emit(("error", exc))
                        except (queue.Full, RuntimeError):
                            pass
                    finally:
                        try:
                            await emit(None)
                        except (queue.Full, RuntimeError):
                            pass

                future = submit_async_task(run_graph())
                deadline = time.monotonic() + MAX_TURN_SECONDS
                try:
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("The chat turn exceeded its time limit.")
                        try:
                            event = events.get(timeout=min(1, remaining))
                        except queue.Empty:
                            if future.done():
                                error = future.exception()
                                if error:
                                    raise error
                                break
                            continue

                        if event is None:
                            break
                        message_chunk, metadata = event
                        if message_chunk == "error":
                            raise metadata

                        if isinstance(message_chunk, ToolMessage):
                            tool_name = getattr(message_chunk, "name", "tool")
                            if tool_status["box"] is None:
                                tool_status["box"] = st.status(
                                    f"Using {tool_name}...", expanded=False
                                )
                            else:
                                tool_status["box"].update(
                                    label=f"Using {tool_name}...", state="running"
                                )

                        if isinstance(message_chunk, AIMessage):
                            text = content_text(message_chunk.content)
                            if text:
                                yield text
                finally:
                    if not future.done():
                        future.cancel()

            ai_message = st.write_stream(response_stream())
            if tool_status["box"] is not None:
                tool_status["box"].update(label="Tool finished", state="complete")

        if not isinstance(ai_message, str):
            ai_message = content_text(ai_message)
        st.session_state.message_history.append(
            {"role": "assistant", "content": ai_message}
        )
    except Exception:
        logger.exception("Chat request failed")
        st.error("The chatbot is temporarily unavailable. Please try again later.")
