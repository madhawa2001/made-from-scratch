import os 
import json
import random
import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        
        self.function_mappings = function_mappings
        
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words
    
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        """Parse the intents file and populate the necessary data structures."""
        # Check if the nltk tokenizer is available, download if not
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK WordNet...")
            nltk.download('wordnet')
            
        lemmatizer = nltk.WordNetLemmatizer()

        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")
            
        try:
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            
            # Debug print to check the loaded data
            print(f"Loaded {len(intents_data.get('intents', []))} intents from file")
            
            # Check if 'intents' key exists
            if 'intents' not in intents_data:
                raise ValueError("Invalid intents file format: missing 'intents' key")
            
            for intent in intents_data['intents']:
                if 'tag' not in intent or 'patterns' not in intent or 'responses' not in intent:
                    print(f"Warning: Intent is missing required fields: {intent}")
                    continue
                    
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                    for pattern in intent['patterns']:
                        pattern_words = self.tokenize_and_lemmatize(pattern)
                        self.vocabulary.extend(pattern_words)
                        self.documents.append((pattern_words, intent['tag']))
            
            if not self.documents:
                raise ValueError("No training documents were generated from intents file")
                
            self.vocabulary = sorted(set(self.vocabulary))
            print(f"Created vocabulary with {len(self.vocabulary)} words")
            print(f"Created {len(self.documents)} training examples")
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in intents file: {self.intents_path}")
    
    def prepare_data(self):
        """Convert the processed documents into training data matrices."""
        if not self.documents:
            raise ValueError("No documents to prepare. Call parse_intents() first.")
            
        bags = []
        indices = []

        for document in self.documents:
            words = document[0] 
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)
        
        print(f"Prepared training data: X shape = {self.X.shape}, y shape = {self.y.shape}")
        
        if len(self.X) == 0 or len(self.y) == 0:
            raise ValueError("Prepared training data is empty")
    
    def train_model(self, batch_size=8, lr=0.001, epochs=100):
        """Train the chatbot model."""
        if self.X is None or self.y is None:
            raise ValueError("Training data not prepared. Call prepare_data() first.")
            
        if len(self.X) == 0:
            raise ValueError("Cannot train on empty dataset")
            
        # Adjust batch size if it's larger than the dataset
        if batch_size > len(self.X):
            batch_size = max(1, len(self.X) // 2)
            print(f"Warning: Batch size adjusted to {batch_size} due to small dataset size")
            
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        """Save the trained model and its dimensions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({
                "input_size": self.X.shape[1], 
                'output_size': len(self.intents),
                'vocabulary': self.vocabulary,
                'intents': self.intents,
                'intents_responses': self.intents_responses
            }, f)
            
        print(f"Model saved to {model_path} and dimensions to {dimensions_path}")
    
    def load_model(self, model_path, dimensions_path):
        """Load a trained model and its dimensions."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if not os.path.exists(dimensions_path):
            raise FileNotFoundError(f"Dimensions file not found: {dimensions_path}")
            
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set model to evaluation mode
        
        # Load the vocabulary and intents
        self.vocabulary = dimensions['vocabulary']
        self.intents = dimensions['intents']
        self.intents_responses = dimensions['intents_responses']
        
        print(f"Model loaded from {model_path}")

    def process_message(self, input_message):
        """Process a user message and return a response."""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
            
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]
        
        # Get confidence score
        confidence = torch.softmax(predictions, dim=1)[0][predicted_class_index].item()
        print(f"Predicted intent: {predicted_intent} (confidence: {confidence:.2f})")

        # Execute function if mapped
        result = None
        if self.function_mappings and predicted_intent in self.function_mappings:
            result = self.function_mappings[predicted_intent]()
            print(f"Executed function for intent '{predicted_intent}': {result}")
        
        # Get a response
        if predicted_intent in self.intents_responses and self.intents_responses[predicted_intent]:
            response = random.choice(self.intents_responses[predicted_intent])
            
            # Replace placeholders if result exists
            if result and isinstance(response, str) and "{result}" in response:
                response = response.replace("{result}", str(result))
                
            return response
        else:
            return "I'm not sure how to respond to that."

def get_stocks():
    stocks = ['APPLE', 'META', 'NVDA', 'GS', 'MSFT']
    return random.sample(stocks, 3)

if __name__ == '__main__':
    # Get intents file path from user
    intents_file = input("Enter the path to your intents JSON file (default: intents.json): ").strip()
    if not intents_file:
        intents_file = 'intents.json'
    
    try:
        # Initialize the assistant with function mappings
        function_mappings = {'stocks': get_stocks}
        assistant = ChatbotAssistant(intents_file, function_mappings=function_mappings)
        
        # Parse intents and prepare data
        print("Parsing intents...")
        assistant.parse_intents()
        
        print("Preparing training data...")
        assistant.prepare_data()
        
        # Train the model
        print("Training model...")
        epochs = int(input("Enter number of training epochs (default: 100): ") or 1000)
        batch_size = int(input("Enter batch size (default: 8): ") or 8)
        learning_rate = float(input("Enter learning rate (default: 0.001): ") or 0.0005)
        
        assistant.train_model(batch_size=batch_size, lr=learning_rate, epochs=epochs)
        
        # Save the model
        print("Saving model...")
        model_path = input("Enter path to save model (default: chatbot_model.pth): ").strip() or "chatbot_model.pth"
        dimensions_path = input("Enter path to save dimensions (default: dimensions.json): ").strip() or "dimensions.json"
        
        assistant.save_model(model_path, dimensions_path)
        
        # Interactive chat loop
        print("\nChatbot is ready! Type '/quit' to exit.")
        while True:
            message = input('You: ')
            if message.lower() == '/quit':
                break
                
            response = assistant.process_message(message)
            print(f"Bot: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
