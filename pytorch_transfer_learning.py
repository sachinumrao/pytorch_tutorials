# Import required libraries
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "Ferrari is a greatest car in the"

def predict_next_word(text):

    import time
    t1 = time.time()
    indexed_tokens = tokenizer.encode(text)

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Set the model in evaluation mode to deactivate the DropOut modules
    model.eval()

    # If you have a GPU, put everything on cuda
    #tokens_tensor = tokens_tensor.to('cuda')
    #model.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # Get the predicted next sub-word
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

    # Print the predicted word
    print(predicted_text)
    t2 = time.time()

    print("Time taken : ", t2-t1)


predict_next_word(text)