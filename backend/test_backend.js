const axios = require("axios");

axios.post("http://localhost:5000/recommend", {
  prompt: "mystery with ghosts and haunting",
})
.then(res => {
  console.log("Recommended Books:");
  console.log(res.data);
})
.catch(err => {
  console.error("Backend Test Failed:", err.message);
});
