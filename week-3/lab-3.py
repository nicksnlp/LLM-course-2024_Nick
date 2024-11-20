import google.generativeai as genai
import configparser
import os  # Added import for os

# Get API key
API_KEY = os.environ.get("API_GEMINI")
genai.configure(api_key=API_KEY)

LLM = "gemini-1.5-flash"
model = genai.GenerativeModel(LLM)

# Read system prompts from config file
prompts = configparser.ConfigParser()
prompts.read('prompts_setup.env')

# Loop through all questions in the QUESTIONS section
#for key in prompts["QUESTIONS"]: Doesn't work
messages = []

# Fetch the question from the config
#question = prompts.get("TEMPLATES", "TOPIC")
#system_prompt = f"Please give me some advice on the following: {question}"
system_prompt = f"Please give me some advice on the following: golf"


# Append the prompt to messages
messages.append(system_prompt)

# Generate response
r = model.generate_content(messages)#.text
print(r)

# Append the response for few-shot prompting
messages.append(r)
