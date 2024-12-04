import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import google.generativeai as genai
    
import pandas as pd
import json


def model_init(model: str, temperature: float):

    dotenv.load_dotenv('.env')
    try:
        model4 = ChatOpenAI(model="gpt-4", temperature=temperature)
        model4o = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=temperature)
        model4om = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=temperature)
        model35s = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=temperature)
        model3o = ChatAnthropic(model="claude-3-opus-20240229", temperature=temperature)
        model3h = ChatAnthropic(model="claude-3-haiku-20240307", temperature=temperature)
        model15p = genai.GenerativeModel("gemini-1.5-pro")

        prompt = ChatPromptTemplate.from_template("{valuation}")
        output_parser = StrOutputParser()

        chain4 = prompt | model4 | output_parser
        chain4o = prompt | model4o | output_parser
        chain4om = prompt | model4om | output_parser
        chain35s = prompt | model35s | output_parser
        chain3o = prompt | model3o | output_parser
        chain3h = prompt | model3h | output_parser
    except Exception as e:
        print(f"Error initializing models: {e}")

    if model == "gpt4":
        return chain4
    elif model == "gpt4o":
        return chain4o
    elif model == "gpt4om":
        return chain4om
    elif model == "claude35s":
        return chain35s
    elif model == "claude3o":
        return chain3o
    elif model == "claude3h":
        return chain3h
    elif model == "gemini15p":
        return model15p
    else:
        return None

def query_model(agents: int, items: int, input:str, model, path: str, type_of_dist: str = "uniform"):

    agents = agents
    items = items
    type_of_dist = type_of_dist

    # print(f"Generating valuations for {agents} agents and {items} items with a {type_of_dist} distribution...")

    with open(path, "w") as file:
        file.write("Input:\n")
        # print("Generating input prompt...")
        file.write(input)
        file.write("\n")
        file.write("Model: GPT4o\n")
        file.write(f"Type of distribution: {type_of_dist}\n")
        file.write("Temperature: 0.7\n")
        file.write("\n")
        file.write("Output:\n")
        # print(f"Querying model {model}...")
        output = model.invoke({"valuation": input})
        file.write(output)
        file.write("\n")
    
    return output

def extract_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find the last occurrence of "json"
    last_occurrence = content.rfind("json")

    quotes_occurrence = content.rfind("```")
    
    if last_occurrence == -1:
        # If "json" is not found, return None or a custom message
        return None
    
    # Extract everything from the last occurrence to the end
    result = content[last_occurrence + 4:quotes_occurrence]

    output_json = json.loads(result)
    output_dict = {int(k): v for k, v in output_json.items()}

    return output_dict