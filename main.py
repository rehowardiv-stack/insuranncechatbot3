import chainlit as cl
from groq import Groq
import os

# Basic CSS overrides injected directly (targets Chainlit chat elements)
# This gives a clean, modern look without needing extra folders/files
CUSTOM_CSS = """
.cl-root, .cl-chat {
    max-width: 1000px !important;
    margin: 0 auto !important;
    font-family: system-ui, -apple-system, sans-serif;
}
.cl-message {
    border-radius: 18px !important;
    padding: 16px 20px !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.1) !important;
    margin-bottom: 16px !important;
}
.cl-user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}
.cl-assistant-message {
    background: #f1f5f9 !important;
    color: #1e293b !important;
}
.cl-input-container {
    border: 2px solid #667eea !important;
    border-radius: 16px !important;
    background: white !important;
    box-shadow: 0 4px 12px rgba(102,126,234,0.1) !important;
}
.cl-btn-send {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: 12px !important;
}
"""

# Groq client – key comes from Railway env variable
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@cl.on_chat_start
async def start():
    # Apply the custom style
    await cl.AppStyle(content=CUSTOM_CSS)
    
    # Welcome message
    await cl.Message(
        content="Hey there! I'm your Groq-powered chatbot. Ask me anything – let's go! 🚀"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Create a streaming message
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Call Groq with streaming
        stream = client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # fast & capable model – change if you prefer another
            messages=[{"role": "user", "content": message.content}],
            temperature=0.7,
            max_tokens=1500,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                await msg.stream_token(token)

        await msg.update()

    except Exception as e:
        await cl.Message(content=f"Sorry, something went wrong: {str(e)}").send()
