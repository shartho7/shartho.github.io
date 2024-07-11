from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "gpt3"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a function for chatting with proper attention mask
def chat(input_text):
    # Encode user input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate response with proper attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
    
    # Decode and print output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Main loop for interactive chatting
print("Welcome to ChatGPT! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("ChatGPT: Goodbye!")
        break
    
    response = chat(user_input)
    print("ChatGPT:", response)
