# Using an open source LLM
# Visit https://ollama.com/ and download and install it
# Then open a terminal and run the command ollama pull llama3.2 then ollama list

from openai import OpenAI

ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
model_name = "llama3.2:latest"

question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{
              "role": "user", 
			  "content": question
		   }]

response = ollama.chat.completions.create(
    model=model_name, 
    messages=messages
)

question = response.choices[0].message.content
print(f"Ollama Question: {question}")

# Ollama gave us a question - Send it back to Ollama and ask for its answer
messages = [{
              "role": "user", 
			  "content": question
		   }]

response = ollama.chat.completions.create(
    model=model_name, 
    messages=messages
)

answer = response.choices[0].message.content
print(f"Ollama Answer: {answer}")
