import requests

response = requests.post(
    "http://127.0.0.1:5001/recommend",
    json={"prompt": "fantasy with dragons and magic"}
)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
