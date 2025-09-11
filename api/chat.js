// api/chat.js
import fetch from "node-fetch"; // Required for Node <18. In Node 18+, fetch is global

export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end(); // Preflight request
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    // Read the request body manually
    let body = "";
    req.on("data", chunk => { body += chunk.toString(); });
    await new Promise(resolve => req.on("end", resolve));

    const { inputs } = JSON.parse(body); // Parse JSON

    if (!inputs) {
      return res.status(400).json({ error: "No input provided" });
    }

    // Call Hugging Face API
    const hfResponse = await fetch(
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

    const data = await hfResponse.json();

    // Return JSON to frontend
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(data));

  } catch (err) {
    // Catch parsing or Hugging Face errors
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: err.message }));
  }
}
