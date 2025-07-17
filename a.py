import json
import pandas as pd
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from typing import Optional
from datasets import Dataset
import os
import pickle
from src.integration import DataIntegrator

CONFIG = {
    'model_name': 'vinai/phobert-base',
    'max_length': 128,
    'train_batch_size': 16,
    'eval_batch_size': 32,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'save_steps': 500,
    'eval_steps': 500,
    'logging_steps': 100,
    'early_stopping_patience': 3,
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'output_dir': '/kaggle/working/models'
}

class VNSLUDataProcessor:
    def __init__(self, tokenizer_name: str = 'vinai/phobert-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.intent_encoder = LabelEncoder()
        self.slot_labels = ['O']  # Start with Outside label
        self.data = []
        
    def load_dataset(self, file_path: str, sample_size: Optional[int] = None):
        """Load VN-SLU dataset with error handling"""
        print(f"Loading dataset from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                        
                        # Sample limiting for testing
                        if sample_size and len(self.data) >= sample_size:
                            print(f"Limited to {sample_size} samples for testing")
                            break
                            
                    except json.JSONDecodeError:
                        print(f"Skipped line {line_num}: JSON decode error")
                        continue
                    except Exception as e:
                        print(f"Error on line {line_num}: {e}")
                        continue
            
            print(f"Loaded {len(self.data)} samples successfully")
            return True
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return False
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def prepare_slot_labels(self):
        """Create BIO labels for slot filling"""
        entity_types = set()
        for item in self.data:
            for entity in item.get('entities', []):
                entity_types.add(entity.get('type', ''))
        
        # Create BIO labels
        slot_labels = ['O']
        for entity_type in sorted(entity_types):
            if entity_type:  # Skip empty types
                slot_labels.extend([f'B-{entity_type}', f'I-{entity_type}'])
        
        self.slot_labels = slot_labels
        print(f"Created {len(self.slot_labels)} slot labels: {self.slot_labels[:10]}...")
        
    def create_intent_dataset(self):
        """Create dataset for intent classification"""
        sentences = []
        intents = []
        
        for item in self.data:
            sentence = item.get('sentence', '').strip()
            intent = item.get('intent', '').strip()
            
            if sentence and intent:
                sentences.append(sentence)
                intents.append(intent)
        
        # Encode intents
        encoded_intents = self.intent_encoder.fit_transform(intents)
        
        print(f"Intent dataset: {len(sentences)} samples")
        print(f"Unique intents: {len(self.intent_encoder.classes_)}")
        print(f"Sample intents: {list(self.intent_encoder.classes_)[:5]}...")
        
        return Dataset.from_dict({
            'text': sentences,
            'labels': encoded_intents
        })
    
    def create_ner_dataset(self):
        """Create dataset for slot filling (NER)"""
        sentences = []
        labels = []
        
        # Create label mappings
        label2id = {label: i for i, label in enumerate(self.slot_labels)}
        
        for item in self.data:
            sentence = item.get('sentence', '').strip()
            entities = item.get('entities', [])
            
            if not sentence:
                continue
            
            # Tokenize sentence
            tokens = self.tokenizer.tokenize(sentence)
            if len(tokens) > CONFIG['max_length'] - 2:  # Account for special tokens
                tokens = tokens[:CONFIG['max_length'] - 2]
            
            # Initialize all tokens as 'O' (Outside)
            token_labels = ['O'] * len(tokens)
            
            # Mark entity tokens with BIO labels
            for entity in entities:
                entity_text = entity.get('filler', '').strip()
                entity_type = entity.get('type', '').strip()
                
                if not entity_text or not entity_type:
                    continue
                
                # Simple token matching (can be improved with better alignment)
                entity_tokens = self.tokenizer.tokenize(entity_text)
                
                # Find entity position in sentence tokens
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        # Mark with BIO labels
                        if i < len(token_labels):
                            token_labels[i] = f'B-{entity_type}'
                        for j in range(1, len(entity_tokens)):
                            if i + j < len(token_labels):
                                token_labels[i + j] = f'I-{entity_type}'
                        break
            
            # Convert to label IDs
            label_ids = [label2id.get(label, 0) for label in token_labels]
            
            sentences.append(tokens)
            labels.append(label_ids)
        
        print(f"NER dataset: {len(sentences)} samples")
        print(f"Sample tokens: {sentences[0][:10] if sentences else 'None'}")
        print(f"Sample labels: {labels[0][:10] if labels else 'None'}")
        
        return Dataset.from_dict({
            'tokens': sentences,
            'labels': labels
        })
    
    def save_encoders(self, output_dir: str):
        """Save label encoders for later use"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save intent encoder
        with open(f"{output_dir}/intent_encoder.pkl", 'wb') as f:
            pickle.dump(self.intent_encoder, f)
        
        # Save slot labels
        with open(f"{output_dir}/slot_labels.json", 'w', encoding='utf-8') as f:
            json.dump(self.slot_labels, f, ensure_ascii=False, indent=2)
        
        print(f"Encoders saved to {output_dir}")

processor = VNSLUDataProcessor()
