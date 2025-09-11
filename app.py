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
    Updated respond function without OAuth requirement
    """
    # Use your HF token from environment variable or hardcode it (not recommended for production)
    hf_token = os.getenv("HF_TOKEN")  # Set this in your Space secrets
    
    if not hf_token:
        return "Error: Hugging Face token not configured"
    
    client = InferenceClient(token=hf_token, model="openai/gpt-oss-20b")

    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""

    try:
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = message_chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content

            response += token
            yield response
    except Exception as e:
        yield f"Error: {str(e)}"

# Create the ChatInterface
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

# Create the main app interface
with gr.Blocks(title="Nails by Navia Bot") as demo:
    chatbot.render()

if __name__ == "__main__":
    demo.launch()
