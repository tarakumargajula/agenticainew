# pip install boto3
import boto3
import json

# Create a Bedrock client (region must support Bedrock, e.g. us-east-1 or us-west-2)
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Define the model
model_id = "amazon.titan-text-lite-v1"

# Titan requires "inputText" for the request body
payload = {
    "inputText": "What is Agentic AI?"
}

# Send the prompt
response = client.invoke_model(
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
    body=json.dumps(payload)  # always send JSON string
)

# Read and decode the response
output = json.loads(response["body"].read())
#print(json.dumps(output, indent=2))
print(output["results"][0]["outputText"])

