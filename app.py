import gradio as gr
from huggingface_hub import InferenceClient
import os

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "microsoft/DialoGPT-medium"  # you can swap this with any chat-capable model
SYSTEM_MESSAGE = (
    "You are Navia, a friendly and knowledgeable nail art specialist. "
    "Help users with nail care, design ideas, and beauty tips. Be creative and enthusiastic!"
)

# -------------------------------
# RESPONSE FUNCTION
# -------------------------------
def respond(message, history, system_message, max_tokens, temperature, top_p):
    """
    Respond to a user message. Supports streaming output.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        yield "❌ Error: Please set HF_TOKEN in your Space secrets."
        return

    try:
        client = InferenceClient(model=MODEL_NAME, token=hf_token)

        # Build OpenAI-style messages
        messages = [{"role": "system", "content": system_message}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": message})

        response = ""

        # Stream response
        for event in client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True
        ):
            delta = event.choices[0].delta.get("content", "")
            if delta:
                response += delta
                yield response

    except Exception as e:
        yield f"❌ Error: {str(e)}"

# -------------------------------
# GRADIO CHAT APP
# -------------------------------
demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="💅 Nails by Navia Bot",
    description="Your personal nail art and beauty assistant!",
    additional_inputs=[
        gr.Textbox(
            value=SYSTEM_MESSAGE,
            label="System Message",
            lines=3
        ),
        gr.Slider(
            minimum=1, maximum=1000, value=512, step=1, label="Max Tokens"
        ),
        gr.Slider(
            minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"
        ),
        gr.Slider(
            minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"
        ),
    ],
    examples=[
        "What are some trendy nail colors for fall?",
        "How can I make my manicure last longer?",
        "Show me some easy nail art ideas for beginners",
        "What's the best way to care for my cuticles?"
    ]
)

# -------------------------------
# LAUNCH APP
# -------------------------------
if __name__ == "__main__":
    demo.launch(
        share=False,
        show_api=True,   # enable API so frontend can call /chat/completions
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )
