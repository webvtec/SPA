export default async function handler(req, res) {
  try {
    const { inputs } = req.body;

    const response = await fetch(
      "https://api-inference.huggingface.co/models/WebVtec/nails-by-navia-bot",
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.HF_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs }),
      }
    );

    const data = await response.json();
    res.status(200).json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}
