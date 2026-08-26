import os
import logging
import chainlit as cl
from openai import OpenAI
from config.settings import (
    CHAINLIT_HOST,
    CHAINLIT_PORT,
    INFERENCE_SERVER_URL,
    REQUEST_TIMEOUT,
    MODEL_ID,
    REWRITE_MODEL_ID,
    ENABLE_Q_REWRITE,
    MAX_TOKENS,
    TEMPERATURE,
)
from agents.registry import agents, determine_agent_type
from models.classification import AgentType

# ------------------------------------------------------------
# Initialize OpenAI-compatible client for vLLM or TensorRT-LLM server
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@cl.cache
def get_session_context():
    """Maintain active agent and conversation history per session."""
    return {"active_agent": "general", "conversation_history": []}


@cl.set_starters
async def set_starters():
    """Welcome screen starter prompts."""
    return [
        cl.Starter(
            label="Email to Professor",
            message=(
                "Help me compose a professional email to my professor requesting an extension "
                "for my term paper due to health issues."
            ),
            icon="/public/icons/email.svg",
        ),
        cl.Starter(
            label="Research Paper Assistant",
            message=(
                "I need help structuring my research paper on climate change impacts. "
                "Can you provide an outline with sections I should include?"
            ),
            icon="/public/icons/research.svg",
        ),
        cl.Starter(
            label="Academic concepts",
            message="Explain the concept of quantum mechanics and its fundamental principles.",
            icon="/public/icons/academic.svg",
        ),
        cl.Starter(
            label="Graduate Admissions Info",
            message=(
                "Where can I find information about graduate admissions requirements for the CS department?"
            ),
            icon="/public/icons/url.svg",
        ),
    ]


def get_llm_client():
    """Get an OpenAI-compatible client for the inference server."""
    return OpenAI(
        api_key="EMPTY",
        base_url=INFERENCE_SERVER_URL,
        timeout=REQUEST_TIMEOUT,
    )


def verify_llm_server():
    """Ping the vLLM/OpenAI-compatible server to ensure readiness."""
    try:
        client = get_llm_client()
        client.models.list()
        logger.info(f"Inference server at {INFERENCE_SERVER_URL} is reachable")
        return True
    except Exception as e:
        logger.error(f"Cannot reach inference server at {INFERENCE_SERVER_URL}: {e}")
        return False


@cl.on_chat_start
async def on_chat_start():
    """At session start, verify the LLM server and warn if unreachable."""
    logger.info("New chat session started")
    if not verify_llm_server():
        await cl.Message(
            content=(
                "Warning: LLM server connection failed. Responses may be delayed "
                "or unavailable. Contact your administrator."
            )
        ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Main message handler: agent selection, optional rewrite, and vLLM inference."""
    user_input = message.content.strip()
    logger.info(f"Received user input: {user_input}")

    context = get_session_context()
    current_agent_type = context["active_agent"]
    current_agent = agents[current_agent_type]

    attachments = getattr(message, "attachments", None)
    has_attach = bool(attachments)

    # Switch to Vision agent if image attached
    if has_attach and current_agent_type != AgentType.VISION:
        current_agent.reset()
        context["active_agent"] = AgentType.VISION
        current_agent_type = AgentType.VISION
        current_agent = agents[AgentType.VISION]
        logger.info("Switched to Vision agent due to image attachment")

    # Optional question rewrite step
    refined_input = user_input
    if ENABLE_Q_REWRITE:
        try:
            rewriter = OpenAI(
                api_key="EMPTY",
                base_url=INFERENCE_SERVER_URL,
                timeout=REQUEST_TIMEOUT,
            )
            rewrite_prompt = (
                "Rewrite the question for clarity without adding new facts:\n\nQuestion: "
                + user_input
            )
            rewrite_resp = rewriter.chat.completions.create(
                model=REWRITE_MODEL_ID,
                messages=[{"role": "user", "content": rewrite_prompt}],
                max_tokens=64,
                temperature=0.2,
            )
            refined_input = rewrite_resp.choices[0].message.content.strip()
            logger.info(f"Question refined to: {refined_input}")
        except Exception as e:
            logger.warning(f"Rewrite failed: {e}; using original input.")

    # Agent detection based on keywords and attachments
    if not current_agent.waiting_for_input and not has_attach:
        lower_ref = refined_input.lower()
        code_kw = ["code", "script", "function", "algorithm", "snippet"]
        email_kw = ["email", "compose", "draft", "extension"]
        if any(w in lower_ref for w in code_kw):
            detected = AgentType.GENERAL
        elif any(w in lower_ref for w in email_kw):
            detected = AgentType.EMAIL
        else:
            detected = determine_agent_type(refined_input, has_attachment=has_attach)
        if detected != current_agent_type:
            current_agent.reset()
            context["active_agent"] = detected
            current_agent_type = detected
            current_agent = agents[detected]
            logger.info(f"Switched to {current_agent.name} agent")

    # Append to conversation history
    context["conversation_history"].append({"role": "user", "content": refined_input})

    # Generate response using the agent
    result = current_agent.process_input(refined_input)

    # Skip interactive prompts
    if result["type"] == "input_request":
        result["type"] = "final_response"

    if result["type"] == "final_response":
        if not verify_llm_server():
            await cl.Message(
                content=(
                    "The LLM server appears to be offline or unreachable. "
                    "Please try again later or contact an administrator."
                )
            ).send()
            return

        msg = cl.Message(content="")
        await msg.send()

        # Stream tokens at moderate rate
        response = await current_agent.get_response(
            refined_input, attachments
        )
        import asyncio
        for char in response:
            await msg.stream_token(char)
            await asyncio.sleep(0.01)
        await msg.update()

        context["conversation_history"].append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    logger.info("Starting Chainlit application with CUDA device %s", CUDA_DEVICE)
    cl.run(host=CHAINLIT_HOST, port=CHAINLIT_PORT)
