import os 
import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

class EnhancedChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(EnhancedChatbotModel, self).__init__()
        
        # Define the layers with configurable hidden sizes
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Activation and regularization
        self.relu = nn.LeakyReLU(0.1)  # Using LeakyReLU instead of regular ReLU
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for hidden_size in hidden_sizes
        ])

    def forward(self, x):
        # Apply each layer with activation, batch normalization and dropout
        for i, (layer, batch_norm) in enumerate(zip(self.layers[:-1], self.batch_norm_layers)):
            x = layer(x)
            if x.size(0) > 1:  # Only apply batch norm when batch size > 1
                x = batch_norm(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # Output layer (no activation or dropout)
        x = self.layers[-1](x)
        return x

class EnhancedChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None, use_tfidf=True, use_stemming=True):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.raw_documents = []  # Store raw text for TF-IDF
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        
        self.function_mappings = function_mappings
        
        self.X = None
        self.y = None
        
        # Enhanced NLP processing
        self.use_tfidf = use_tfidf
        self.use_stemming = use_stemming
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer() if use_stemming else None
        
        # Threshold for confidence
        self.confidence_threshold = 0.3

    def preprocess_text(self, text):
        """Preprocess text with tokenization, lemmatization and optional stemming."""
        # Tokenize
        words = nltk.word_tokenize(text.lower())
        
        # Remove punctuation and normalize
        words = [word for word in words if word.isalnum()]
        
        # Lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        # Stem if enabled
        if self.use_stemming:
            words = [self.stemmer.stem(word) for word in words]
            
        return words
    
    def bag_of_words(self, words):
        """Convert a list of words to a bag-of-words representation."""
        return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        """Parse the intents file and populate the necessary data structures."""
        # Download required NLTK resources
        for resource in ['punkt', 'wordnet']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading NLTK {resource}...")
                nltk.download(resource)
            
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")
            
        try:
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
            
            print(f"Loaded {len(intents_data.get('intents', []))} intents from file")
            
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
                        # Store the raw pattern for TF-IDF
                        self.raw_documents.append(pattern)
                        
                        # Process for bag-of-words
                        pattern_words = self.preprocess_text(pattern)
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
        
        if self.use_tfidf:
            # Use TF-IDF vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit features to prevent overfitting
                ngram_range=(1, 2)  # Use both unigrams and bigrams
            )
            
            # Train the vectorizer on all patterns
            tfidf_matrix = self.vectorizer.fit_transform(self.raw_documents)
            
            # Convert to feature vectors for each document
            X_tfidf = tfidf_matrix.toarray()
            self.X = X_tfidf
            
            print(f"Created TF-IDF features with shape: {self.X.shape}")
        else:
            # Use traditional bag of words
            bags = []
            for document in self.documents:
                words = document[0] 
                bag = self.bag_of_words(words)
                bags.append(bag)
                
            self.X = np.array(bags)
            
        # Create target labels
        indices = [self.intents.index(document[1]) for document in self.documents]
        self.y = np.array(indices)
        
        print(f"Prepared training data: X shape = {self.X.shape}, y shape = {self.y.shape}")
        
        if len(self.X) == 0 or len(self.y) == 0:
            raise ValueError("Prepared training data is empty")
    
    def train_model(self, batch_size=8, lr=0.001, epochs=200, hidden_layers=None):
        """Train the chatbot model with enhanced architecture."""
        if self.X is None or self.y is None:
            raise ValueError("Training data not prepared. Call prepare_data() first.")
            
        if len(self.X) == 0:
            raise ValueError("Cannot train on empty dataset")
            
        # Default hidden layer sizes if none provided
        if hidden_layers is None:
            hidden_layers = [128, 64]
        
        # Adjust batch size if it's larger than the dataset
        if batch_size > len(self.X):
            batch_size = max(1, len(self.X) // 2)
            print(f"Warning: Batch size adjusted to {batch_size} due to small dataset size")
            
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create model with specified architecture
        input_size = self.X.shape[1]
        output_size = len(self.intents)
        
        self.model = EnhancedChatbotModel(
            input_size=input_size,
            hidden_sizes=hidden_layers,
            output_size=output_size,
            dropout_rate=0.5
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Learning rate scheduler - reduce learning rate every 20 epochs
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        # Keep track of best validation loss for early stopping
        best_loss = float('inf')
        patience = 10  # Number of epochs to wait before early stopping
        patience_counter = 0
        
        # Training loop with validation
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training on {train_size} samples, validating on {val_size} samples")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Learning rate scheduler step
            scheduler.step()
            
            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save the best model
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(best_model_state)
                    break

    def save_model(self, model_path, dimensions_path):
        """Save the trained model and its dimensions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        torch.save(self.model.state_dict(), model_path)

        # Save configuration
        config = {
            "input_size": self.X.shape[1], 
            'output_size': len(self.intents),
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'intents_responses': self.intents_responses,
            'use_tfidf': self.use_tfidf,
            'use_stemming': self.use_stemming
        }
        
        # If using TF-IDF, save the vectorizer
        if self.use_tfidf and self.vectorizer:
            if not os.path.exists('model'):
                os.makedirs('model')
            import pickle
            with open('model/vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
                
            config['vectorizer_path'] = 'model/vectorizer.pkl'

        with open(dimensions_path, 'w') as f:
            json.dump(config, f)
            
        print(f"Model saved to {model_path} and configuration to {dimensions_path}")
    
    def load_model(self, model_path, dimensions_path):
        """Load a trained model and its configuration."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if not os.path.exists(dimensions_path):
            raise FileNotFoundError(f"Configuration file not found: {dimensions_path}")
            
        with open(dimensions_path, 'r') as f:
            config = json.load(f)
        
        # Load model configuration
        input_size = config['input_size'] 
        output_size = config['output_size']
        
        # Determine hidden layers (default if not in saved config)
        hidden_layers = config.get('hidden_layers', [128, 64])
        
        # Create model with the same architecture
        self.model = EnhancedChatbotModel(
            input_size=input_size,
            hidden_sizes=hidden_layers,
            output_size=output_size
        )
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load vocabulary and intents
        self.vocabulary = config['vocabulary']
        self.intents = config['intents']
        self.intents_responses = config['intents_responses']
        self.use_tfidf = config.get('use_tfidf', False)
        self.use_stemming = config.get('use_stemming', False)
        
        # Load vectorizer if TF-IDF was used
        if self.use_tfidf and 'vectorizer_path' in config:
            import pickle
            with open(config['vectorizer_path'], 'rb') as f:
                self.vectorizer = pickle.load(f)
        
        print(f"Model loaded from {model_path}")

    def process_message(self, input_message):
        """Process a user message and return a response."""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # Preprocess the input message
        if self.use_tfidf and self.vectorizer:
            # Use TF-IDF vectorization
            X = self.vectorizer.transform([input_message]).toarray()
        else:
            # Use bag of words
            words = self.preprocess_text(input_message)
            X = [self.bag_of_words(words)]
            
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        # Get probabilities
        probabilities = torch.softmax(predictions, dim=1)[0]
        max_prob, predicted_class_index = torch.max(probabilities, dim=0)
        
        # Convert to Python values
        confidence = max_prob.item()
        predicted_class_index = predicted_class_index.item()
        
        # Get intent and confidence
        predicted_intent = self.intents[predicted_class_index]
        print(f"Predicted intent: {predicted_intent} (confidence: {confidence:.2f})")
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return "I'm not sure I understand. Could you rephrase your question?"

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
        # Ask for advanced options
        print("\nAdvanced options:")
        use_tfidf = input("Use TF-IDF instead of bag-of-words? (y/n, default: y): ").lower() != 'n'
        use_stemming = input("Use stemming in addition to lemmatization? (y/n, default: y): ").lower() != 'n'
        
        # Initialize the assistant
        function_mappings = {'stocks': get_stocks}
        assistant = EnhancedChatbotAssistant(
            intents_file, 
            function_mappings=function_mappings,
            use_tfidf=use_tfidf,
            use_stemming=use_stemming
        )
        
        # Parse intents and prepare data
        print("\nParsing intents...")
        assistant.parse_intents()
        
        print("Preparing training data...")
        assistant.prepare_data()
        
        # Get training parameters
        print("\nTraining parameters:")
        epochs = int(input("Enter number of training epochs (default: 200): ") or 200)
        batch_size = int(input("Enter batch size (default: 8): ") or 8)
        learning_rate = float(input("Enter learning rate (default: 0.0005): ") or 0.0005)
        
        # Get model architecture
        print("\nModel architecture:")
        num_hidden_layers = int(input("Number of hidden layers (default: 3): ") or 3)
        hidden_layers = []
        for i in range(num_hidden_layers):
            size = int(input(f"Size of hidden layer {i+1} (default: {128 if i==0 else 64}): ") 
                      or (128 if i==0 else 64))
            hidden_layers.append(size)
        
        # Train the model
        print("\nTraining model...")
        assistant.train_model(
            batch_size=batch_size, 
            lr=learning_rate, 
            epochs=epochs,
            hidden_layers=hidden_layers
        )
        
        # Save the model
        print("\nSaving model...")
        if not os.path.exists('model'):
            os.makedirs('model')
        model_path = input("Enter path to save model (default: model/enhanced_chatbot.pth): ").strip() or "model/enhanced_chatbot.pth"
        dimensions_path = input("Enter path to save config (default: model/config.json): ").strip() or "model/config.json"
        
        assistant.save_model(model_path, dimensions_path)
        
        # Configure runtime parameters
        print("\nRuntime parameters:")
        assistant.confidence_threshold = float(input("Confidence threshold (0.0-1.0, default: 0.3): ") or 0.3)
        
        # Interactive chat loop
        print("\nEnhanced chatbot is ready! Type '/quit' to exit.")
        while True:
            message = input('\nYou: ')
            if message.lower() == '/quit':
                break
                
            response = assistant.process_message(message)
            print(f"Bot: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()