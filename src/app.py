import chainlit as cl
import logging
from config.settings import (
    CHAINLIT_HOST,
    CHAINLIT_PORT,
    INFERENCE_SERVER_URL,
    REQUEST_TIMEOUT,
    MODEL_ID,
    REWRITE_MODEL_ID,
    ENABLE_Q_REWRITE
)
from agents.registry import agents, determine_agent_type
from models.classification import AgentType
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Session context to store the active agent for each user session
@cl.cache
def get_session_context():
    return {
        "active_agent": "general",  # Default to general agent
        "conversation_history": []
    }

@cl.set_starters
async def set_starters():
    """Define university-related starter suggestions for the welcome screen."""
    return [
        cl.Starter(
            label="Email to Professor",
            message="Help me compose a professional email to my professor requesting an extension for my term paper due to health issues.",
            icon="/public/icons/email.svg",
        ),
        cl.Starter(
            label="Research Paper Assistant",
            message="I need help structuring my research paper on climate change impacts. Can you provide a outline with sections I should include?",
            icon="/public/icons/research.svg",
        ),
        cl.Starter(
            label="Academic related concepts",
            message="Explain the concept of quantum mechanics and its fundamental principles.",
            icon="/public/icons/academic.svg",
        ),
        cl.Starter(
            label="Redirect me to ?",
            message="Where can I find information about graduate admissions requirements for the Computer Science department?",
            icon="/public/icons/url.svg",
        )
    ]

# Verify LLM server connection
def verify_llm_server():
    """Check if the LLM server is available"""
    try:
        test_client = OpenAI(
            api_key="EMPTY", 
            base_url=INFERENCE_SERVER_URL,
            timeout=5.0
        )
        # Quick ping to check server
        test_client.models.list()
        logger.info("LLM server is reachable")
        return True
    except Exception as e:
        logger.error(f"Cannot reach LLM server at {INFERENCE_SERVER_URL}: {str(e)}")
        return False

@cl.on_chat_start
async def on_chat_start():
    """Initialize the session and display welcome message"""
    logger.info("New chat session started")
    
    # Verify LLM server at session start
    is_server_ready = verify_llm_server()
    if not is_server_ready:
        await cl.Message(
            content="Warning: LLM server connection failed. Responses may be delayed or unavailable. Please contact an administrator."
        ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    """
    Handle incoming user messages by:
    1. Determining the appropriate agent
    2. Collecting necessary inputs
    3. Generating responses
    """
    user_input = message.content.strip()
    logger.info(f"Received user input: {user_input}")
    
    # Get the session context and current active agent
    context = get_session_context()
    current_agent_type = context["active_agent"]
    current_agent = agents[current_agent_type]
    
    # Check attachments
    attachments = getattr(message, "attachments", None)
    has_attach = bool(attachments)

    # If we have an image attachment, force Vision agent
    if has_attach and current_agent_type != AgentType.VISION:
        current_agent.reset()
        context["active_agent"] = AgentType.VISION
        current_agent_type = AgentType.VISION
        current_agent = agents[AgentType.VISION]
        logger.info("Switched to Vision agent due to image attachment")

    # OPTIONAL: Improve / refine the question before classification using a lightweight LLM call
    refined_input = user_input
    if ENABLE_Q_REWRITE:
        try:
            # Using vLLM endpoint (or same model) as zero-shot question rewriter
            rewriter = OpenAI(api_key="EMPTY", base_url=INFERENCE_SERVER_URL, timeout=REQUEST_TIMEOUT)
            rewrite_prompt = (
                "You are an assistant that rewrites student questions to be clearer and more complete "
                "without adding new facts. Return ONLY the rewritten question.\n\nQuestion: " + user_input)
            rewrite_resp = rewriter.chat.completions.create(
                model=REWRITE_MODEL_ID,
                messages=[{"role": "user", "content": rewrite_prompt}],
                max_tokens=64,
                temperature=0.2
            )
            refined_input = rewrite_resp.choices[0].message.content.strip()
            logger.info(f"Question refined to: {refined_input}")
        except Exception as e:
            logger.warning(f"Could not refine question: {e}. Proceeding with original input.")

    # Otherwise check if new conversation direction using refined input
    if not current_agent.waiting_for_input and not has_attach:
        lower_ref = refined_input.lower()
        code_keywords = ["code", "program", "script", "function", "algorithm", "source", "snippet"]
        email_keywords = ["email", "mail", "compose", "draft", "send", "extension", "meeting"]

        if any(word in lower_ref for word in code_keywords):
            detected_agent_type = AgentType.GENERAL  # Use General agent to answer coding questions
        elif any(word in lower_ref for word in email_keywords):
            detected_agent_type = AgentType.EMAIL
        else:
            detected_agent_type = determine_agent_type(refined_input, has_attachment=has_attach)
        if detected_agent_type != current_agent_type:
            # Reset the previous agent before switching
            current_agent.reset()
            
            # Switch to the newly detected agent
            context["active_agent"] = detected_agent_type
            current_agent_type = detected_agent_type
            current_agent = agents[current_agent_type]
            logger.info(f"Switched to {current_agent.name} agent")
    
    # Add message to conversation history (store refined input to help context)
    context["conversation_history"].append({"role": "user", "content": refined_input})
    
    # Process the input with the current agent
    result = current_agent.process_input(refined_input)

    # --- NEW LOGIC: skip interactive follow-up questions ---
    if result["type"] == "input_request":
        logger.info("Skipping extra input collection. Proceeding to final response directly.")
        result["type"] = "final_response"

    if result["type"] == "final_response":
        # First verify server is available before attempting response
        is_server_ready = verify_llm_server()
        if not is_server_ready:
            await cl.Message(
                content="The LLM server appears to be offline or unreachable. Please try again later or contact an administrator."
            ).send()
            return

        # Agent has all it needs, generate a response
        msg = cl.Message(content="")  # thinking indicator
        await msg.send()

        # Generate full response
        response = await current_agent.get_response(refined_input, attachments)

        # Stream the response to the user at a moderate speed
        import asyncio
        for char in response:
            await msg.stream_token(char)
            await asyncio.sleep(0.01)  # adjust speed here (0.01 ≈ 100 chars/sec)
        await msg.update()  # Ensure final update

        # Add response to conversation history
        context["conversation_history"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    logger.info("Starting Chainlit application")
    cl.run(host=CHAINLIT_HOST, port=CHAINLIT_PORT)