from transformers import AutoTokenizer, AutoModelForSequenceClassification
from azureml.core import Workspace, Model

# Load pre-trained model from Hugging Face
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("finished")
# Save the model locally
model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")
print("finished")
