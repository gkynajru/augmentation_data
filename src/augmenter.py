#!/usr/bin/env python3
"""
Vietnamese SLU Data Augmentation Core Module
============================================

This module provides the core functionality for augmenting Vietnamese SLU datasets
with quality control and entity preservation.
"""

import json
import random
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime
import yaml

class VietnameseSLUAugmenter:
    """
    Core augmentation engine for Vietnamese SLU data
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the augmenter with configuration"""
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize vocabularies from config
        self.device_synonyms = self.config['vocabularies']['devices']
        self.location_synonyms = self.config['vocabularies']['locations']
        self.action_synonyms = self.config['vocabularies']['actions']
        self.politeness_prefixes = self.config['vocabularies']['politeness']['prefixes']
        self.politeness_suffixes = self.config['vocabularies']['politeness']['suffixes']
        self.time_variations = self.config['vocabularies']['time_expressions']
        
        # Quality control settings
        self.min_length_ratio = self.config['quality_control']['min_length_ratio']
        self.max_length_ratio = self.config['quality_control']['max_length_ratio']
        
        # Tracking
        self.generation_stats = defaultdict(int)
        self.quality_issues = []
        self.processed_samples = 0
        
        self.logger.info("Vietnamese SLU Augmenter initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_dataset(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze dataset to identify rare intents and entities for targeted augmentation
        """
        self.logger.info(f"Analyzing dataset with {len(data)} samples")
        
        # Count intent frequencies
        intent_counts = Counter(item.get('intent', '') for item in data)
        
        # Count entity type frequencies
        entity_counts = Counter()
        for item in data:
            for entity in item.get('entities', []):
                entity_type = entity.get('type', '')
                if entity_type:
                    entity_counts[entity_type] += 1
        
        # Identify rare categories
        min_intent_freq = self.config['augmentation']['min_intent_frequency']
        min_entity_freq = self.config['augmentation']['min_entity_frequency']
        
        rare_intents = set(intent for intent, count in intent_counts.items() 
                          if count < min_intent_freq)
        rare_entities = set(entity_type for entity_type, count in entity_counts.items() 
                           if count < min_entity_freq)
        
        analysis = {
            'total_samples': len(data),
            'intent_distribution': dict(intent_counts),
            'entity_distribution': dict(entity_counts),
            'rare_intents': list(rare_intents),
            'rare_entities': list(rare_entities),
            'unique_intents': len(intent_counts),
            'unique_entities': len(entity_counts),
            'augmentation_targets': {
                'intent_targets': len(rare_intents),
                'entity_targets': len(rare_entities)
            }
        }
        
        self.logger.info(f"Analysis complete: {len(rare_intents)} rare intents, {len(rare_entities)} rare entities")
        return analysis
    
    def should_augment_sample(self, item: Dict, rare_intents: set, rare_entities: set) -> bool:
        """
        Determine if a sample should be augmented based on rarity criteria
        """
        intent = item.get('intent', '')
        entities = item.get('entities', [])
        
        # Check if intent is rare
        if self.config['augmentation']['augment_rare_intents'] and intent in rare_intents:
            return True
        
        # Check if any entity is rare
        if self.config['augmentation']['augment_rare_entities']:
            for entity in entities:
                if entity.get('type', '') in rare_entities:
                    return True
        
        return False
    
    def generate_augmentations(self, sentence: str, intent: str, entities: List[Dict], 
                             num_variations: Optional[int] = None) -> List[Dict]:
        """
        Generate augmented versions of a sample
        """
        if num_variations is None:
            num_variations = self.config['augmentation']['max_variations_per_sample']
        
        augmentations = []
        methods = self.config['augmentation']['sample_augmentation_methods']
        
        for i in range(min(num_variations, len(methods))):
            method = methods[i % len(methods)]
            
            try:
                aug_sentence, aug_entities, metadata = self._apply_augmentation_method(
                    sentence, entities, method
                )
                
                if self._quality_check(sentence, aug_sentence, entities, aug_entities):
                    augmentations.append({
                        'original_sentence': sentence,
                        'augmented_sentence': aug_sentence,
                        'original_intent': intent,
                        'augmented_intent': intent,  # Intent preserved
                        'original_entities': entities,
                        'augmented_entities': aug_entities,
                        'method': method,
                        'metadata': metadata,
                        'quality_passed': True,
                        'timestamp': datetime.now().isoformat()
                    })
                    self.generation_stats[method] += 1
                else:
                    self.quality_issues.append({
                        'original_sentence': sentence,
                        'augmented_sentence': aug_sentence,
                        'method': method,
                        'reason': 'quality_check_failed',
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                self.logger.error(f"Error in method {method}: {e}")
                self.quality_issues.append({
                    'original_sentence': sentence,
                    'method': method,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return augmentations
    
    def _apply_augmentation_method(self, sentence: str, entities: List[Dict], 
                                  method: str) -> Tuple[str, List[Dict], Dict]:
        """
        Apply a specific augmentation method
        """
        metadata = {'method': method, 'original_length': len(sentence.split())}
        
        if method == "synonym_replacement":
            aug_sentence, aug_entities = self._replace_synonyms(sentence, entities)
            
        elif method == "politeness_addition":
            aug_sentence, aug_entities = self._add_politeness_markers(sentence, entities)
            
        elif method == "sentence_restructuring":
            aug_sentence, aug_entities = self._restructure_sentence(sentence, entities)
            
        elif method == "time_variation":
            aug_sentence, aug_entities = self._vary_time_expressions(sentence, entities)
            
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        metadata['augmented_length'] = len(aug_sentence.split())
        metadata['length_ratio'] = metadata['augmented_length'] / metadata['original_length']
        metadata['entity_count_preserved'] = len(aug_entities) == len(entities)
        
        return aug_sentence, aug_entities, metadata
    
    def _replace_synonyms(self, sentence: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Replace entities and actions with Vietnamese synonyms
        """
        aug_sentence = sentence
        aug_entities = []
        
        # Replace entity synonyms while preserving boundaries
        for entity in entities:
            entity_text = entity.get('filler', '').strip()
            entity_type = entity.get('type', '')
            
            replacement = self._find_entity_synonym(entity_text, entity_type)
            
            if replacement and replacement != entity_text and replacement in aug_sentence:
                # Only replace if the synonym actually appears in sentence context
                aug_sentence = aug_sentence.replace(entity_text, replacement, 1)
                new_entity = entity.copy()
                new_entity['filler'] = replacement
                aug_entities.append(new_entity)
            else:
                aug_entities.append(entity.copy())
        
        # Replace action words
        aug_sentence = self._replace_action_words(aug_sentence)
        
        return aug_sentence, aug_entities
    
    def _find_entity_synonym(self, entity_text: str, entity_type: str) -> Optional[str]:
        """
        Find appropriate synonym for an entity based on type
        """
        entity_lower = entity_text.lower()
        
        # Device synonyms
        if 'device' in entity_type.lower():
            for original, synonyms in self.device_synonyms.items():
                if original in entity_lower:
                    return random.choice(synonyms)
                # Check reverse mapping
                if entity_lower in synonyms:
                    return original
        
        # Location synonyms
        if any(keyword in entity_type.lower() for keyword in ['space', 'location', 'room']):
            for original, synonyms in self.location_synonyms.items():
                if original in entity_lower:
                    return random.choice(synonyms)
                if entity_lower in synonyms:
                    return original
        
        return None
    
    def _replace_action_words(self, sentence: str) -> str:
        """
        Replace action words with synonyms
        """
        words = sentence.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.action_synonyms:
                synonyms = self.action_synonyms[word_lower]
                words[i] = random.choice(synonyms)
                break  # Only replace one action word to maintain naturalness
        
        return " ".join(words)
    
    def _add_politeness_markers(self, sentence: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Add Vietnamese politeness markers
        """
        aug_sentence = sentence
        
        # Add prefix (40% chance)
        if random.random() < 0.4:
            prefix = random.choice(self.politeness_prefixes)
            aug_sentence = f"{prefix} {aug_sentence}"
        
        # Add suffix (60% chance) - more common in Vietnamese
        if random.random() < 0.6:
            suffix = random.choice(self.politeness_suffixes)
            aug_sentence = f"{aug_sentence} {suffix}"
        
        # Entities remain unchanged as we only add to sentence boundaries
        return aug_sentence, entities.copy()
    
    def _restructure_sentence(self, sentence: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Restructure sentence while preserving entity annotations
        """
        # Simple restructuring: move time/location expressions
        
        # Pattern 1: Move time to beginning
        time_pattern = r'(.*?)(lúc|vào|tại) ([^,\s]+(?:\s+[^,\s]+)*?)(\s|$|,)'
        match = re.search(time_pattern, sentence)
        if match:
            before, time_marker, time_expr, after = match.groups()
            restructured = f"{time_marker} {time_expr}, {before.strip()}{after}"
            return restructured, entities.copy()
        
        # Pattern 2: Move location to beginning  
        location_pattern = r'(.*?)(trong|ở|tại) ([^,\s]+(?:\s+[^,\s]+)*?)(\s|$|,)'
        match = re.search(location_pattern, sentence)
        if match:
            before, loc_marker, location, after = match.groups()
            restructured = f"{loc_marker} {location}, {before.strip()}{after}"
            return restructured, entities.copy()
        
        # If no restructuring possible, return original
        return sentence, entities.copy()
    
    def _vary_time_expressions(self, sentence: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Vary time expressions while updating entity annotations
        """
        aug_sentence = sentence
        aug_entities = []
        
        for entity in entities:
            entity_text = entity.get('filler', '').strip()
            entity_type = entity.get('type', '')
            
            # Check if this is a time-related entity
            if any(time_word in entity_type.lower() for time_word in ['time', 'duration', 'delay']):
                replacement = self._get_time_variation(entity_text)
                if replacement:
                    aug_sentence = aug_sentence.replace(entity_text, replacement, 1)
                    new_entity = entity.copy()
                    new_entity['filler'] = replacement
                    aug_entities.append(new_entity)
                else:
                    aug_entities.append(entity.copy())
            else:
                aug_entities.append(entity.copy())
        
        # Also vary general time words not marked as entities
        for original, variations in self.time_variations.items():
            if original in aug_sentence.lower():
                replacement = random.choice(variations)
                aug_sentence = re.sub(
                    r'\b' + re.escape(original) + r'\b', 
                    replacement, 
                    aug_sentence, 
                    count=1,
                    flags=re.IGNORECASE
                )
                break
        
        return aug_sentence, aug_entities
    
    def _get_time_variation(self, time_text: str) -> Optional[str]:
        """
        Get variation for time expression
        """
        time_lower = time_text.lower()
        
        for original, variations in self.time_variations.items():
            if original in time_lower:
                return random.choice(variations)
        
        return None
    
    def _quality_check(self, original: str, augmented: str, 
                      original_entities: List[Dict], augmented_entities: List[Dict]) -> bool:
        """
        Comprehensive quality control for augmented samples
        """
        
        # Basic checks
        if not augmented or augmented.strip() == "":
            return False
        
        if augmented == original:
            return False  # No actual augmentation
        
        # Length ratio check
        orig_len = len(original.split())
        aug_len = len(augmented.split())
        
        if orig_len == 0:
            return False
        
        length_ratio = aug_len / orig_len
        if length_ratio < self.min_length_ratio or length_ratio > self.max_length_ratio:
            return False
        
        # Entity preservation check
        if self.config['quality_control']['check_entity_preservation']:
            if len(augmented_entities) != len(original_entities):
                return False
            
            # Check that entity types are preserved
            orig_types = sorted([e.get('type', '') for e in original_entities])
            aug_types = sorted([e.get('type', '') for e in augmented_entities])
            if orig_types != aug_types:
                return False
        
        # Vietnamese character check
        if self.config['quality_control']['require_vietnamese_characters']:
            if not self._has_vietnamese_characters(augmented):
                return False
        
        # Basic coherence check
        if not self._basic_coherence_check(augmented):
            return False
        
        return True
    
    def _has_vietnamese_characters(self, text: str) -> bool:
        """
        Check if text contains Vietnamese characters
        """
        vietnamese_pattern = r'[aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵđ]'
        return bool(re.search(vietnamese_pattern, text.lower()))
    
    def _basic_coherence_check(self, text: str) -> bool:
        """
        Basic check for text coherence
        """
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 3:
            word_counts = Counter(words)
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) // 2:  # More than half are same word
                return False
        
        # Check for reasonable sentence structure (basic heuristic)
        if len(words) > 20:  # Very long sentences might be malformed
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get augmentation statistics
        """
        total_generated = sum(self.generation_stats.values())
        total_failed = len(self.quality_issues)
        
        return {
            'processed_samples': self.processed_samples,
            'total_generated': total_generated,
            'total_failed': total_failed,
            'success_rate': total_generated / (total_generated + total_failed) if (total_generated + total_failed) > 0 else 0,
            'method_breakdown': dict(self.generation_stats),
            'quality_issue_count': total_failed,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_statistics(self):
        """Reset tracking statistics"""
        self.generation_stats = defaultdict(int)
        self.quality_issues = []
        self.processed_samples = 0
        self.logger.info("Statistics reset")