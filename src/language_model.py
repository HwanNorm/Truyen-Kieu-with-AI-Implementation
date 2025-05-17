import numpy as np
import random
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict

class NGramModel:
    """Statistical n-gram model for Vietnamese text generation"""
    
    def __init__(self, n=3, smoothing=0.01):
        self.n = n
        self.smoothing = smoothing
        self.context_counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)
        self.vocabulary = set()
    
    def train(self, verses: List[str]):
        """Train the model on a corpus of verses"""
        # Add special tokens and build vocabulary
        for verse in verses:
            # Add start/end tokens
            processed_verse = ['<s>'] * (self.n - 1) + list(verse) + ['</s>']
            self.vocabulary.update(processed_verse)
            
            # Count n-grams
            for i in range(len(processed_verse) - self.n + 1):
                context = tuple(processed_verse[i:i+self.n-1])
                next_char = processed_verse[i+self.n-1]
                self.context_counts[context][next_char] += 1
                self.context_totals[context] += 1
    
    def generate(self, start_text: str, max_length: int = 100) -> str:
        """Generate text continuing from the start_text"""
        if not self.context_counts:
            raise ValueError("Model has not been trained yet")
            
        # Prepare the initial context
        current_seq = ['<s>'] * (self.n - 1) + list(start_text)
        result = list(start_text)
        
        # Generate until we reach max length or end token
        while len(result) < max_length:
            context = tuple(current_seq[-(self.n-1):])
            
            # If context not seen or reached end, stop
            if context not in self.context_counts or result[-1] == '</s>':
                break
                
            # Get probabilities for next character
            possible_chars = list(self.context_counts[context].keys())
            counts = np.array([self.context_counts[context][char] for char in possible_chars])
            
            # Apply smoothing
            probs = (counts + self.smoothing) / (self.context_totals[context] + 
                                                self.smoothing * len(self.vocabulary))
            
            # Normalize probabilities to ensure they sum to 1
            probs = probs / probs.sum()
            # Sample next character
            next_char = np.random.choice(possible_chars, p=probs)
            
            # Add to result if not end token
            if next_char != '</s>':
                result.append(next_char)
            else:
                break
                
            current_seq.append(next_char)
            
        return ''.join(result)
    
    def save(self, filepath: str):
        """Save the model to a file"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'smoothing': self.smoothing,
                'context_counts': dict(self.context_counts),
                'context_totals': dict(self.context_totals),
                'vocabulary': self.vocabulary
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a model from a file"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        model = cls(n=data['n'], smoothing=data['smoothing'])
        model.context_counts = defaultdict(Counter, data['context_counts'])
        model.context_totals = defaultdict(int, data['context_totals'])
        model.vocabulary = data['vocabulary']
        
        return model


class KieuLSTM(nn.Module):
    """LSTM-based language model for Truyện Kiều"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.2):
        super(KieuLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """Forward pass through the network"""
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
            
        output = self.dropout(output)
        
        # Final layer
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int):
        """Initialize hidden state"""
        weight = next(self.parameters()).data
        
        return (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())


class KieuTokenizer:
    """Simple character-level tokenizer for Vietnamese text"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def fit(self, texts: List[str]):
        """Build the vocabulary from texts"""
        # Collect all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)
            
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        vocabulary = special_tokens + sorted(all_chars)
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(vocabulary)
        
        # Special token indices
        self.pad_token_id = self.char_to_idx['<pad>']
        self.unk_token_id = self.char_to_idx['<unk>']
        self.start_token_id = self.char_to_idx['<s>']
        self.end_token_id = self.char_to_idx['</s>']
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        return [self.char_to_idx.get(char, self.unk_token_id) for char in text]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        # Filter out special tokens
        filtered_ids = [id for id in ids if id != self.pad_token_id 
                        and id != self.start_token_id and id != self.end_token_id]
        
        return ''.join([self.idx_to_char.get(id, '<unk>') for id in filtered_ids])
    

class KieuLanguageModelTrainer:
    """Class for training language models on Truyện Kiều"""
    
    def __init__(self, model_type='ngram', n=3, smoothing=0.01, 
                 embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.2):
        self.model_type = model_type
        self.model = None
        self.tokenizer = KieuTokenizer()
        
        # Model hyperparameters
        self.n = n
        self.smoothing = smoothing
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
    def prepare_data(self, verses: List[str]):
        """Prepare data for training without actual training"""
        self.tokenizer.fit(verses)
        
        if self.model_type == 'lstm':
            self.model = KieuLSTM(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        elif self.model_type == 'ngram':
            self.model = NGramModel(n=self.n, smoothing=self.smoothing)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, verses: List[str], epochs: int = 5, batch_size: int = 32, 
              learning_rate: float = 0.001):
        """Train the language model"""
        if self.model is None:
            self.prepare_data(verses)
            
        if self.model_type == 'ngram':
            # Training for NGramModel
            self.model.train(verses)
            print("NGram model training complete")
            
        elif self.model_type == 'lstm':
            # Training data for LSTM
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            
            # Prepare dataset
            encoded_verses = [self.tokenizer.encode(verse) for verse in verses]
            
            # Create input-target pairs (input: verse[:-1], target: verse[1:])
            inputs = []
            targets = []
            
            for verse in encoded_verses:
                if len(verse) < 2:  # Skip very short verses
                    continue
                inputs.append(verse[:-1])
                targets.append(verse[1:])
            
            # Training loop
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            
            # Convert to tensor dataset
            from torch.utils.data import TensorDataset, DataLoader
            
            # Pad sequences
            from torch.nn.utils.rnn import pad_sequence
            padded_inputs = pad_sequence([torch.tensor(x) for x in inputs], 
                                         batch_first=True, 
                                         padding_value=self.tokenizer.pad_token_id)
            padded_targets = pad_sequence([torch.tensor(x) for x in targets], 
                                          batch_first=True, 
                                          padding_value=self.tokenizer.pad_token_id)
            
            dataset = TensorDataset(padded_inputs, padded_targets)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                
                for batch_inputs, batch_targets in dataloader:
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output, _ = self.model(batch_inputs)
                    
                    # Reshape for loss calculation
                    output = output.view(-1, self.tokenizer.vocab_size)
                    batch_targets = batch_targets.view(-1)
                    
                    # Calculate loss
                    loss = criterion(output, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            print("LSTM model training complete")
    
    def train_epoch(self, verses: List[str], batch_size: int = 32, learning_rate: float = 0.001):
        """Train the model for a single epoch (for incremental training)"""
        if self.model_type != 'lstm':
            raise ValueError("This method is only for LSTM models")
            
        # Ensure we have a model and tokenizer initialized
        if not hasattr(self, 'tokenizer') or not self.tokenizer.vocab_size:
            self.prepare_data(verses)
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Prepare dataset
        encoded_verses = [self.tokenizer.encode(verse) for verse in verses]
        
        # Create input-target pairs
        inputs = []
        targets = []
        
        for verse in encoded_verses:
            if len(verse) < 2:  # Skip very short verses
                continue
            inputs.append(verse[:-1])
            targets.append(verse[1:])
        
        # Set up optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Create data loader
        from torch.utils.data import TensorDataset, DataLoader
        from torch.nn.utils.rnn import pad_sequence
        
        # Pad sequences
        padded_inputs = pad_sequence([torch.tensor(x) for x in inputs], 
                                    batch_first=True, 
                                    padding_value=self.tokenizer.pad_token_id)
        padded_targets = pad_sequence([torch.tensor(x) for x in targets], 
                                    batch_first=True, 
                                    padding_value=self.tokenizer.pad_token_id)
        
        dataset = TensorDataset(padded_inputs, padded_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train for one epoch
        self.model.train()
        total_loss = 0
        
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.model(batch_inputs)
            
            # Reshape for loss calculation
            output = output.view(-1, self.tokenizer.vocab_size)
            batch_targets = batch_targets.view(-1)
            
            # Calculate loss
            loss = criterion(output, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    

    def train_with_early_stopping(self, verses: List[str], epochs: int = 100, 
                                batch_size: int = 128, learning_rate: float = 0.001,
                                patience: int = 5, validation_split: float = 0.1):
        """Train with early stopping based on validation loss"""
        if self.model_type != 'lstm':
            raise ValueError("This method is only for LSTM models")
            
        # Ensure we have a model and tokenizer
        if self.model is None:
            self.prepare_data(verses)
        
        # Split data into training and validation sets
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(verses)
        split_idx = int(len(verses) * (1 - validation_split))
        train_verses = verses[:split_idx]
        val_verses = verses[split_idx:]
        
        print(f"Training on {len(train_verses)} verses, validating on {len(val_verses)} verses")
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Process training data
        train_encoded = [self.tokenizer.encode(verse) for verse in train_verses]
        train_inputs = []
        train_targets = []
        
        for verse in train_encoded:
            if len(verse) < 2:  # Skip very short verses
                continue
            train_inputs.append(verse[:-1])
            train_targets.append(verse[1:])
        
        # Process validation data
        val_encoded = [self.tokenizer.encode(verse) for verse in val_verses]
        val_inputs = []
        val_targets = []
        
        for verse in val_encoded:
            if len(verse) < 2:  # Skip very short verses
                continue
            val_inputs.append(verse[:-1])
            val_targets.append(verse[1:])
        
        # Create data loaders
        from torch.utils.data import TensorDataset, DataLoader
        from torch.nn.utils.rnn import pad_sequence
        
        # Pad sequences
        train_padded_inputs = pad_sequence([torch.tensor(x) for x in train_inputs], 
                                        batch_first=True, 
                                        padding_value=self.tokenizer.pad_token_id)
        train_padded_targets = pad_sequence([torch.tensor(x) for x in train_targets], 
                                        batch_first=True, 
                                        padding_value=self.tokenizer.pad_token_id)
        
        val_padded_inputs = pad_sequence([torch.tensor(x) for x in val_inputs], 
                                        batch_first=True, 
                                        padding_value=self.tokenizer.pad_token_id)
        val_padded_targets = pad_sequence([torch.tensor(x) for x in val_targets], 
                                        batch_first=True, 
                                        padding_value=self.tokenizer.pad_token_id)
        
        train_dataset = TensorDataset(train_padded_inputs, train_padded_targets)
        val_dataset = TensorDataset(val_padded_inputs, val_padded_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        # Training loop with early stopping
        for epoch in range(epochs):
            # Train for one epoch
            self.model.train()
            train_loss = 0
            
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output, _ = self.model(batch_inputs)
                
                # Reshape for loss calculation
                output = output.view(-1, self.tokenizer.vocab_size)
                batch_targets = batch_targets.view(-1)
                
                # Calculate loss
                loss = criterion(output, batch_targets)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    # Forward pass
                    output, _ = self.model(batch_inputs)
                    
                    # Reshape for loss calculation
                    output = output.view(-1, self.tokenizer.vocab_size)
                    batch_targets = batch_targets.view(-1)
                    
                    # Calculate loss
                    loss = criterion(output, batch_targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Check if this is the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = {
                    'model_state_dict': self.model.state_dict(),
                    'tokenizer': {
                        'char_to_idx': self.tokenizer.char_to_idx,
                        'idx_to_char': self.tokenizer.idx_to_char,
                        'vocab_size': self.tokenizer.vocab_size,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'unk_token_id': self.tokenizer.unk_token_id,
                        'start_token_id': self.tokenizer.start_token_id,
                        'end_token_id': self.tokenizer.end_token_id
                    },
                    'params': {
                        'embedding_dim': self.embedding_dim,
                        'hidden_dim': self.hidden_dim,
                        'num_layers': self.num_layers,
                        'dropout': self.dropout
                    },
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Check early stopping condition
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load the best model
        if best_model:
            self.model.load_state_dict(best_model['model_state_dict'])
            print(f"Loaded best model from epoch {best_model['epoch']+1} with validation loss {best_model['val_loss']:.4f}")
            
            # Return best model state for saving
            return best_model
        
        return None

    def generate_with_postprocessing(self, start_text: str, max_length: int = 100) -> str:
        """Generate text and apply Vietnamese poetic post-processing"""
        # Generate raw text with the model
        raw_text = self.generate(start_text, max_length)
        
        # If raw_text is too short, return as is
        if len(raw_text) < 10:
            return raw_text
        
        # Import the post-processor
        from .verse_generator import KieuVerseGenerator
        
        # Create a temporary generator that we'll use just for post-processing
        # We don't need to load a model since we'll only use the post-processing methods
        verse_generator = object.__new__(KieuVerseGenerator)
        verse_generator._init_vietnamese_resources()
        
        # Apply poetic constraints
        processed_text = verse_generator.apply_poetic_constraints(raw_text)
        
        return processed_text
    
    def generate(self, start_text: str, max_length: int = 100) -> str:
        """Generate text using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        if self.model_type == 'ngram':
            return self.model.generate(start_text, max_length)
            
        elif self.model_type == 'lstm':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
            
            # Encode start text
            input_ids = torch.tensor([self.tokenizer.encode(start_text)]).to(device)
            
            # Initial hidden state
            hidden = None
            
            # List to store output IDs
            output_ids = list(input_ids[0].cpu().numpy())
            
            # Generate one character at a time
            with torch.no_grad():
                for _ in range(max_length):
                    # Get predictions
                    output, hidden = self.model(input_ids, hidden)
                    
                    # Get last prediction
                    last_output = output[0, -1, :]
                    
                    # Apply temperature sampling
                    temperature = 0.8
                    probs = torch.softmax(last_output / temperature, dim=0)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # Stop if end token
                    if next_token == self.tokenizer.end_token_id:
                        break
                        
                    # Append to output
                    output_ids.append(next_token)
                    
                    # Update input for next iteration
                    input_ids = torch.tensor([[next_token]]).to(device)
            
            # Decode output
            return self.tokenizer.decode(output_ids)
    
    def save(self, filepath: str):
        """Save the model and tokenizer"""
        if self.model is None:
            raise ValueError("No model to save")
            
        if self.model_type == 'ngram':
            self.model.save(filepath)
        elif self.model_type == 'lstm':
            # Save model state dict and tokenizer
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer': {
                    'char_to_idx': self.tokenizer.char_to_idx,
                    'idx_to_char': self.tokenizer.idx_to_char,
                    'vocab_size': self.tokenizer.vocab_size,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'unk_token_id': self.tokenizer.unk_token_id,
                    'start_token_id': self.tokenizer.start_token_id,
                    'end_token_id': self.tokenizer.end_token_id
                },
                'params': {
                    'embedding_dim': self.embedding_dim,
                    'hidden_dim': self.hidden_dim,
                    'num_layers': self.num_layers,
                    'dropout': self.dropout
                }
            }, filepath)
    
    @classmethod
    def load(cls, filepath: str, model_type: str):
        """Load a saved model"""
        instance = cls(model_type=model_type)
        
        if model_type == 'ngram':
            instance.model = NGramModel.load(filepath)
        elif model_type == 'lstm':
            # Load saved state
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            
            # Restore tokenizer
            instance.tokenizer = KieuTokenizer()
            instance.tokenizer.char_to_idx = checkpoint['tokenizer']['char_to_idx']
            instance.tokenizer.idx_to_char = checkpoint['tokenizer']['idx_to_char']
            instance.tokenizer.vocab_size = checkpoint['tokenizer']['vocab_size']
            instance.tokenizer.pad_token_id = checkpoint['tokenizer']['pad_token_id']
            instance.tokenizer.unk_token_id = checkpoint['tokenizer']['unk_token_id']
            instance.tokenizer.start_token_id = checkpoint['tokenizer']['start_token_id']
            instance.tokenizer.end_token_id = checkpoint['tokenizer']['end_token_id']
            
            # Restore parameters
            instance.embedding_dim = checkpoint['params']['embedding_dim']
            instance.hidden_dim = checkpoint['params']['hidden_dim']
            instance.num_layers = checkpoint['params']['num_layers']
            instance.dropout = checkpoint['params']['dropout']
            
            # Create model
            instance.model = KieuLSTM(
                vocab_size=instance.tokenizer.vocab_size,
                embedding_dim=instance.embedding_dim,
                hidden_dim=instance.hidden_dim,
                num_layers=instance.num_layers,
                dropout=instance.dropout
            )
            
            # Load state dict
            instance.model.load_state_dict(checkpoint['model_state_dict'])
            instance.model.eval()
        
        return instance


# Transformer-based model for Vietnamese
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers import TextDataset, DataCollatorForLanguageModeling
    
    def fine_tune_transformer(model_name="vinai/phobert-base", 
                             poem_file_path="data/truyen_kieu.txt",
                             output_dir="models/kieu_transformer"):
        """Fine-tune a pre-trained Vietnamese transformer model on Truyện Kiều"""
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=poem_file_path,
            block_size=128  # Context window size
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Not using masked language modeling
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return model, tokenizer
    
    class TransformerKieuModel:
        """Wrapper for a fine-tuned transformer model on Truyện Kiều"""
        
        def __init__(self, model_path="models/kieu_transformer"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Ensure we have a padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def generate(self, prompt, max_length=100, num_return_sequences=1):
            """Generate text from a prompt"""
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                temperature=0.8,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # Decode and return
            generated_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                              for ids in output]
            
            return generated_texts
except ImportError:
    # If transformers not installed
    pass


if __name__ == "__main__":
    # Simple test
    print("Testing Language Model Module")
    
    # Example verses
    example_verses = [
        "Trăm năm trong cõi người ta,",
        "Chữ tài chữ mệnh khéo là ghét nhau.",
        "Trải qua một cuộc bể dâu,",
        "Những điều trông thấy mà đau đớn lòng."
    ]
    
    # Test NGram model
    print("\nTesting NGram Model")
    trainer_ngram = KieuLanguageModelTrainer(model_type='ngram', n=2)
    trainer_ngram.train(example_verses)
    
    # Generate from NGram
    start_text = "Trăm năm"
    generated_text = trainer_ngram.generate(start_text, max_length=30)
    print(f"Generated text from '{start_text}':")
    print(generated_text)
    
    # Test LSTM model (minimal for demo)
    if torch.cuda.is_available():
        print("\nTesting LSTM Model")
        trainer_lstm = KieuLanguageModelTrainer(model_type='lstm')
        trainer_lstm.train(example_verses, epochs=2)
        
        # Generate from LSTM
        generated_text = trainer_lstm.generate(start_text, max_length=30)
        print(f"Generated text from '{start_text}':")
        print(generated_text)