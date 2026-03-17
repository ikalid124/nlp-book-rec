require("dotenv").config();

const express = require("express");
const axios = require("axios");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

// Endpoint to call the Python microservice
app.post("/recommend", async (req, res) => {
  try {
    const response = await axios.post(`${process.env.ML_SERVICE_URL}/recommend`, {
      prompt: req.body.prompt,
    });
    res.json(response.data);
  } catch (error) {
    console.error("Error contacting Python service:", error.message);
    res.status(500).json({ error: "Recommendation service failed" });
  }
});

app.listen(PORT, () => {
  console.log(`Node.js backend running at http://localhost:${PORT}`);
});
