import gradio as gr
from huggingface_hub import InferenceClient
import os

def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    Respond function with proper error handling
    """
    # Use environment variable for HF token
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        yield "❌ Error: Please set HF_TOKEN in your Space secrets"
        return
    
    try:
        client = InferenceClient(token=hf_token, model="microsoft/DialoGPT-medium")
        
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history
        for msg in history:
            messages.append(msg)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        response = ""
        
        # Try to get streaming response
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if hasattr(message_chunk, 'choices') and len(message_chunk.choices) > 0:
                delta = message_chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    response += delta.content
                    yield response
                    
    except Exception as e:
        yield f"❌ Error: {str(e)}"

# Create the ChatInterface with API enabled
demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="💅 Nails by Navia Bot",
    description="Your personal nail art and beauty assistant!",
    additional_inputs=[
        gr.Textbox(
            value="You are Navia, a friendly and knowledgeable nail art specialist. Help users with nail care, design ideas, and beauty tips. Be creative and enthusiastic!",
            label="System Message",
            lines=3
        ),
        gr.Slider(
            minimum=1,
            maximum=1000,
            value=512,
            step=1,
            label="Max Tokens"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.7,
            step=0.1,
            label="Temperature"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p"
        ),
    ],
    examples=[
        "What are some trendy nail colors for fall?",
        "How can I make my manicure last longer?",
        "Show me some easy nail art ideas for beginners",
        "What's the best way to care for my cuticles?"
    ]
)

# IMPORTANT: Launch with API enabled
if __name__ == "__main__":
    demo.launch(
        share=False,
        show_api=True,  # This enables the API
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        debug=True  # Enable debug mode
    )
