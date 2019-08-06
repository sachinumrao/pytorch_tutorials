# Import required libraries
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()
import time
t1 = time.time()
def predict_next_word(indexed_tokens):

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # If you have a GPU, put everything on cuda
    #tokens_tensor = tokens_tensor.to('cuda')
    #model.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # Get the predicted next sub-word
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    return predicted_index
 
# Encode a text inputs
text = "Steve Jobs was a bad"
pred_len = 15
pred_id = []
indexed_tokens = tokenizer.encode(text)
for j in range(pred_len):
    print("Prediction Word : ", j+1)
    pred_index = predict_next_word(indexed_tokens)
    pred_id.append(pred_index)
    indexed_tokens.append(pred_index)

# print("Indexed Tokens Type : ", type(indexed_tokens))
# print("Predicted Tokens Type : ", type(pred_id))
#exit()

predicted_text = tokenizer.decode(indexed_tokens)
print(predicted_text)
t2 = time.time()
print("Time taken : ", t2-t1)