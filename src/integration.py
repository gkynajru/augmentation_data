#!/usr/bin/env python3
"""
Integration Module
==================

This module handles integration of augmented data with training pipelines.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional, Union
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

class DataIntegrator:
    """
    Handles integration of augmented data into training datasets
    """
    
    def __init__(self, config: dict = None):
        """Initialize the data integrator"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Integration strategy
        self.strategy = self.config.get('strategy', 'balanced')
        self.preserve_original = self.config.get('preserve_original_data', True)
        self.add_metadata = self.config.get('add_augmentation_metadata', True)
        
    def load_original_dataset(self, file_path: str) -> List[Dict]:
        """
        Load original dataset from JSONL file
        """
        dataset = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dataset.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error parsing line: {e}")
        
        return dataset
    
    def analyze_distribution(self, dataset: List[Dict]) -> Dict[str, Any]:
        """
        Analyze intent and entity distribution in dataset
        """
        intent_counts = Counter()
        entity_counts = Counter()
        entity_intent_pairs = Counter()
        
        for sample in dataset:
            intent = sample.get('intent', 'unknown')
            intent_counts[intent] += 1
            
            for entity in sample.get('entities', []):
                entity_type = entity.get('type', 'unknown')
                entity_counts[entity_type] += 1
                entity_intent_pairs[(entity_type, intent)] += 1
        
        return {
            'total_samples': len(dataset),
            'intent_distribution': dict(intent_counts),
            'entity_distribution': dict(entity_counts),
            'entity_intent_pairs': dict(entity_intent_pairs),
            'unique_intents': len(intent_counts),
            'unique_entities': len(entity_counts)
        }
    
    def balance_dataset(self, original: List[Dict], augmented: List[Dict], 
                       target_distribution: Dict[str, int] = None) -> List[Dict]:
        """
        Balance the dataset according to strategy
        """
        if self.strategy == 'all':
            # Use all augmented samples
            return augmented
        
        elif self.strategy == 'balanced':
            # Balance based on intent/entity distribution
            original_dist = self.analyze_distribution(original)
            
            # Group augmented samples by intent
            augmented_by_intent = defaultdict(list)
            for sample in augmented:
                augmented_by_intent[sample.get('intent', 'unknown')].append(sample)
            
            balanced = []
            
            # Calculate target numbers
            if not target_distribution:
                # Default: try to balance all intents to similar counts
                max_count = max(original_dist['intent_distribution'].values())
                target_distribution = {}
                
                for intent, count in original_dist['intent_distribution'].items():
                    # Don't over-augment already common intents
                    if count < max_count * 0.5:  # If less than 50% of max
                        target_distribution[intent] = min(
                            max_count - count,  # How many we need
                            len(augmented_by_intent[intent])  # How many we have
                        )
                    else:
                        target_distribution[intent] = 0
            
            # Select samples
            for intent, target_count in target_distribution.items():
                available = augmented_by_intent[intent]
                if available:
                    # Take up to target_count samples
                    selected = available[:target_count]
                    balanced.extend(selected)
            
            return balanced
        
        elif self.strategy == 'selective':
            # Only augment rare categories
            original_dist = self.analyze_distribution(original)
            
            # Define rarity threshold
            total_samples = original_dist['total_samples']
            rarity_threshold = total_samples * 0.05  # 5% of total
            
            # Find rare intents
            rare_intents = {
                intent for intent, count in original_dist['intent_distribution'].items()
                if count < rarity_threshold
            }
            
            # Find rare entities
            rare_entities = {
                entity for entity, count in original_dist['entity_distribution'].items()
                if count < rarity_threshold
            }
            
            # Select augmented samples that help rare categories
            selective = []
            for sample in augmented:
                intent = sample.get('intent', 'unknown')
                entities = [e.get('type', 'unknown') for e in sample.get('entities', [])]
                
                # Include if it has rare intent or rare entity
                if intent in rare_intents or any(e in rare_entities for e in entities):
                    selective.append(sample)
            
            return selective
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def merge_datasets(self, original: List[Dict], augmented: List[Dict]) -> List[Dict]:
        """
        Merge original and augmented datasets
        """
        merged = []
        
        # Add original data if configured
        if self.preserve_original:
            for sample in original:
                if self.add_metadata:
                    sample['metadata'] = {
                        'source': 'original',
                        'augmented': False
                    }
                merged.append(sample)
        
        # Add augmented data
        for sample in augmented:
            if self.add_metadata and 'metadata' not in sample:
                sample['metadata'] = {
                    'source': 'augmentation',
                    'augmented': True
                }
            merged.append(sample)
        
        return merged
    
    def deduplicate(self, dataset: List[Dict]) -> List[Dict]:
        """
        Remove duplicate samples based on sentence
        """
        seen_sentences = set()
        deduplicated = []
        
        for sample in dataset:
            sentence = sample.get('sentence', '').strip().lower()
            if sentence and sentence not in seen_sentences:
                seen_sentences.add(sentence)
                deduplicated.append(sample)
        
        self.logger.info(f"Removed {len(dataset) - len(deduplicated)} duplicates")
        return deduplicated
    
    def integrate_augmented_data(self, original_path: str, augmented_data: List[Dict],
                               output_path: str = None) -> Dict[str, Any]:
        """
        Main integration function
        """
        # Load original dataset
        self.logger.info(f"Loading original dataset from {original_path}")
        original = self.load_original_dataset(original_path)
        
        # Analyze distributions
        original_analysis = self.analyze_distribution(original)
        augmented_analysis = self.analyze_distribution(augmented_data)
        
        self.logger.info(f"Original dataset: {original_analysis['total_samples']} samples")
        self.logger.info(f"Augmented data: {augmented_analysis['total_samples']} samples")
        
        # Apply balancing strategy
        balanced_augmented = self.balance_dataset(original, augmented_data)
        self.logger.info(f"Selected {len(balanced_augmented)} augmented samples using '{self.strategy}' strategy")
        
        # Merge datasets
        merged = self.merge_datasets(original, balanced_augmented)
        
        # Deduplicate
        final_dataset = self.deduplicate(merged)
        
        # Analyze final distribution
        final_analysis = self.analyze_distribution(final_dataset)
        
        # Save if output path provided
        if output_path:
            self.save_dataset(final_dataset, output_path)
        
        # Create integration report
        report = {
            'original_analysis': original_analysis,
            'augmented_analysis': augmented_analysis,
            'final_analysis': final_analysis,
            'integration_summary': {
                'original_samples': len(original),
                'augmented_samples': len(augmented_data),
                'selected_augmented': len(balanced_augmented),
                'final_samples': len(final_dataset),
                'duplicates_removed': len(merged) - len(final_dataset),
                'strategy_used': self.strategy
            }
        }
        
        return report
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """
        Save dataset in various formats
        """
        output_format = self.config.get('output_format', 'jsonl')
        
        if output_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in dataset:
                    # Remove metadata if configured
                    if not self.add_metadata and 'metadata' in sample:
                        sample = sample.copy()
                        sample.pop('metadata', None)
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        elif output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        elif output_format == 'csv':
            # Convert to flat format for CSV
            flat_data = []
            for sample in dataset:
                flat_sample = {
                    'sentence': sample.get('sentence', ''),
                    'intent': sample.get('intent', ''),
                    'entities': json.dumps(sample.get('entities', []), ensure_ascii=False)
                }
                if self.add_metadata:
                    flat_sample['source'] = sample.get('metadata', {}).get('source', 'unknown')
                flat_data.append(flat_sample)
            
            df = pd.DataFrame(flat_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Saved {len(dataset)} samples to {output_path}")
    
    def load_augmented_dataset(self, processor, file_path: str):
        """
        Load augmented dataset directly into a processor/training pipeline
        
        This is a helper function for integration with existing training code
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        sample = json.loads(line.strip())
                        # Add to processor - adjust based on your processor's API
                        if hasattr(processor, 'data'):
                            processor.data.append(sample)
                        elif hasattr(processor, 'add_sample'):
                            processor.add_sample(sample)
                        else:
                            # Fallback - try common methods
                            try:
                                processor.append(sample)
                            except:
                                self.logger.error(f"Cannot add sample to processor")
                                raise
                elif file_path.endswith('.json'):
                    data = json.load(f)
                    for sample in data:
                        if hasattr(processor, 'data'):
                            processor.data.append(sample)
                        elif hasattr(processor, 'add_sample'):
                            processor.add_sample(sample)
            
            self.logger.info(f"Successfully loaded augmented data into processor")
            
        except Exception as e:
            self.logger.error(f"Error loading augmented dataset: {e}")
            raise


def load_augmented_dataset(processor, file_path: str):
    """
    Convenience function to load augmented dataset into a processor
    
    Usage:
        from src.integration import load_augmented_dataset
        load_augmented_dataset(processor, 'data/output/final/final_dataset.jsonl')
    """
    integrator = DataIntegrator()
    integrator.load_augmented_dataset(processor, file_path)