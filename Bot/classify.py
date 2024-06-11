# from huggingface_hub import hf_hub_download
# from llama_cpp import Llama

# ## Download the GGUF model
# model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# model_file = "mistral-7b-instruct-v0.2.Q2_K.gguf" # this is the specific model file we'll use in this example. It's a 4-bit quant, but other levels of quantization are available in the model repo if preferred
# model_path = hf_hub_download(model_name, filename=model_file)

# ## Instantiate model from downloaded file
# llm = Llama(
#     model_path=model_path,
#     n_ctx=16000,  # Context length to use
#     n_threads=32,            # Number of CPU threads to use
#     n_gpu_layers=0        # Number of model layers to offload to GPU
# )

# ## Generation kwargs
# generation_kwargs = {
#     "max_tokens":100,
#     "stop": ["/s"],
#     "echo":False, # Echo the prompt in the output
#     "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
# }

# ## Run inference
# prompt = "Inquiry: Job Openings\nCategory: [Sales/General/HR Enquiry]\nCompany: IndiaNIC. Get the best category"
# res = llm(prompt, **generation_kwargs) # Res is a dictionary

# ## Unpack and the generated text from the LLM response dictionary and print it
# print(res["choices"][0]["text"])


from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def get_sentiment_label(text):
    # Load your model and tokenizer from the local directory
    model_path = "E:/Internship/ChatBot/Bot/sentiment-analysis"  # Specify the path to your model directory
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Define the sentiment analysis pipeline using your local model and tokenizer
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Perform sentiment analysis on the provided text
    result = sentiment_pipeline(text)

    # Extract sentiment label
    sentiment_label = result[0]['label']

    return sentiment_label
