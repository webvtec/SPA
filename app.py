import gradio as gr
from transformers import pipeline

# Load a lightweight model
pipe = pipeline("text-generation", model="microsoft/phi-3-mini-4k-instruct")

def chat_api(message, history=[]):
    """
    message: user input
    history: list of past (user, bot) messages, optional
    """
    # Build the prompt (keep lightweight, no big histories)
    prompt = f"User: {message}\nAssistant:"
    response = pipe(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )[0]["generated_text"]

    # Extract only the assistant's part
    reply = response.split("Assistant:")[-1].strip()

    # ✅ Gradio expects a list/tuple if outputs="text"
    return [reply]

# Gradio app: exposes an API but no heavy UI
demo = gr.Interface(
    fn=chat_api,
    inputs=["text"],
    outputs="text",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
