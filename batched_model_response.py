import numpy

def model_init(model: str, temperature: float):
    dotenv.load_dotenv('.env', override=True)
    try:
        # OpenAI Models
        model4 = ChatOpenAI(model="gpt-4", temperature=temperature)
        model4o = ChatOpenAI(model="gpt-4o", temperature=temperature)
        
        # Anthropic Models
        model35s = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=temperature)
        
        # Google Vertex AI (Gemini)
        model15p = ChatVertexAI(model_name="gemini-1.5-pro", temperature=temperature)

        # Create chains
        prompt = ChatPromptTemplate.from_template("{valuation}")
        output_parser = StrOutputParser()

        return {
            "gpt4": prompt | model4 | output_parser,
            "gpt4o": prompt | model4o | output_parser,
            "claude35s": prompt | model35s | output_parser,
            "gemini15p": prompt | model15p | output_parser
        }
    except Exception as e:
        print(f"Error initializing models: {e}")
        return None
    
from typing import List

def batch_query_model(inputs: List[str], 
                     model_chain, 
                     paths: List[str], 
                     valuations: List[numpy.ndarray],
                     agents: int, 
                     items: int,
                     type_of_dist: str = "uniform",
                     prompt_type: str = "zero-shot0"):
    
    # Prepare batch inputs
    batch_inputs = [{"valuation": input} for input in inputs]
    
    # Process batch
    try:
        outputs = model_chain.batch(batch_inputs)
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return None

    # Write results to files
    for input_text, output_text, path, valuation in zip(inputs, outputs, paths, valuations):
        with open(path, "w") as f:
            f.write(f"Input:\n{input_text}\n")
            f.write(f"Model: {model_chain.model_name}\n")
            f.write(f"Agents: {agents}, Items: {items}\n")
            f.write(f"Distribution: {type_of_dist}\n")
            f.write(f"Temperature: 0.7\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"Valuation Table:\n{valuation}\n")
            f.write(f"Output:\n{output_text}\n")
    
    return outputs