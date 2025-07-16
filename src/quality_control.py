#!/usr/bin/env python3
"""
Quality Control Module
======================

This module provides quality control functions for Vietnamese SLU augmented data.
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

class QualityController:
    """
    Quality control for augmented Vietnamese SLU data
    """
    
    def __init__(self, config: dict = None):
        """Initialize quality controller with configuration"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.min_length_ratio = self.config.get('min_length_ratio', 0.7)
        self.max_length_ratio = self.config.get('max_length_ratio', 2.0)
        self.min_quality_score = self.config.get('min_quality_score', 3.0)
        self.min_naturalness_score = self.config.get('min_naturalness_score', 3.0)
        
        # Vietnamese character set
        self.vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
        
        # Quality issue tracking
        self.quality_issues = []
        
    def check_length_ratio(self, original: str, augmented: str) -> Tuple[bool, Optional[str]]:
        """
        Check if augmented text length is within acceptable range
        """
        if not original or not augmented:
            return False, "Empty text"
            
        ratio = len(augmented) / len(original)
        
        if ratio < self.min_length_ratio:
            return False, f"Too short (ratio: {ratio:.2f})"
        elif ratio > self.max_length_ratio:
            return False, f"Too long (ratio: {ratio:.2f})"
        
        return True, None
    
    def check_vietnamese_characters(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text contains Vietnamese characters
        """
        text_lower = text.lower()
        if not any(char in text_lower for char in self.vietnamese_chars):
            return False, "No Vietnamese characters found"
        return True, None
    
    def check_entity_preservation(self, original_entities: List[Dict], 
                                augmented_entities: List[Dict]) -> Tuple[bool, Optional[str]]:
        """
        Check if entities are properly preserved
        """
        # Check entity count
        if len(original_entities) != len(augmented_entities):
            return False, f"Entity count mismatch: {len(original_entities)} vs {len(augmented_entities)}"
        
        # Check entity types
        orig_types = [e.get('type', '') for e in original_entities]
        aug_types = [e.get('type', '') for e in augmented_entities]
        
        if orig_types != aug_types:
            return False, "Entity types don't match"
        
        # Check if all entities have fillers
        for i, entity in enumerate(augmented_entities):
            if not entity.get('filler'):
                return False, f"Entity {i} missing filler"
        
        return True, None
    
    def check_entity_boundaries(self, sentence: str, entities: List[Dict]) -> Tuple[bool, Optional[str]]:
        """
        Check if entity fillers exist in the sentence
        """
        for entity in entities:
            filler = entity.get('filler', '')
            if filler and filler not in sentence:
                return False, f"Entity '{filler}' not found in sentence"
        
        return True, None
    
    def check_basic_grammar(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Basic Vietnamese grammar checks
        """
        # Check for repeated words
        words = text.split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and words[i] not in ['của', 'và', 'là', 'có']:
                return False, f"Repeated word: '{words[i]}'"
        
        # Check for proper spacing
        if '  ' in text:
            return False, "Multiple spaces detected"
        
        # Check for orphaned punctuation
        if re.search(r'\s[,\.!?]', text):
            return False, "Orphaned punctuation"
        
        return True, None
    
    def check_intent_consistency(self, original_intent: str, augmented_intent: str) -> Tuple[bool, Optional[str]]:
        """
        Check if intent is preserved
        """
        if original_intent != augmented_intent:
            return False, f"Intent changed: '{original_intent}' -> '{augmented_intent}'"
        return True, None
    
    def validate_augmentation(self, original: Dict, augmented: Dict) -> Dict[str, any]:
        """
        Comprehensive validation of an augmented sample
        """
        results = {
            'valid': True,
            'issues': [],
            'checks': {}
        }
        
        # Extract data
        orig_sentence = original.get('sentence', '')
        orig_intent = original.get('intent', '')
        orig_entities = original.get('entities', [])
        
        aug_sentence = augmented.get('sentence', '')
        aug_intent = augmented.get('intent', '')
        aug_entities = augmented.get('entities', [])
        
        # Run all checks
        checks = [
            ('length_ratio', self.check_length_ratio(orig_sentence, aug_sentence)),
            ('vietnamese_chars', self.check_vietnamese_characters(aug_sentence)),
            ('entity_preservation', self.check_entity_preservation(orig_entities, aug_entities)),
            ('entity_boundaries', self.check_entity_boundaries(aug_sentence, aug_entities)),
            ('basic_grammar', self.check_basic_grammar(aug_sentence)),
            ('intent_consistency', self.check_intent_consistency(orig_intent, aug_intent))
        ]
        
        # Process results
        for check_name, (passed, issue) in checks:
            results['checks'][check_name] = passed
            if not passed:
                results['valid'] = False
                results['issues'].append(f"{check_name}: {issue}")
        
        return results
    
    def check_dataset_quality(self, dataset: List[Dict]) -> Dict[str, any]:
        """
        Analyze quality of entire dataset
        """
        total_samples = len(dataset)
        quality_stats = {
            'total_samples': total_samples,
            'passed_all_checks': 0,
            'failed_checks': Counter(),
            'issues_by_type': Counter(),
            'intent_distribution': Counter(),
            'entity_distribution': Counter(),
            'avg_length_ratio': 0,
            'samples_with_vietnamese': 0
        }
        
        length_ratios = []
        
        for sample in dataset:
            # Validate each sample
            if 'original' in sample and 'augmented' in sample:
                validation = self.validate_augmentation(sample['original'], sample['augmented'])
            else:
                # Direct validation for final format
                validation = {
                    'valid': True,
                    'checks': {
                        'vietnamese_chars': self.check_vietnamese_characters(sample.get('sentence', ''))[0],
                        'entity_boundaries': self.check_entity_boundaries(
                            sample.get('sentence', ''),
                            sample.get('entities', [])
                        )[0]
                    }
                }
            
            if validation['valid']:
                quality_stats['passed_all_checks'] += 1
            else:
                for check, passed in validation['checks'].items():
                    if not passed:
                        quality_stats['failed_checks'][check] += 1
            
            # Collect statistics
            quality_stats['intent_distribution'][sample.get('intent', 'unknown')] += 1
            
            for entity in sample.get('entities', []):
                quality_stats['entity_distribution'][entity.get('type', 'unknown')] += 1
            
            # Vietnamese character check
            if self.check_vietnamese_characters(sample.get('sentence', ''))[0]:
                quality_stats['samples_with_vietnamese'] += 1
        
        # Calculate percentages
        quality_stats['quality_rate'] = quality_stats['passed_all_checks'] / total_samples if total_samples > 0 else 0
        quality_stats['vietnamese_rate'] = quality_stats['samples_with_vietnamese'] / total_samples if total_samples > 0 else 0
        
        return quality_stats
    
    def generate_quality_report(self, dataset: List[Dict], output_path: str = None) -> str:
        """
        Generate detailed quality report
        """
        stats = self.check_dataset_quality(dataset)
        
        report = []
        report.append("=" * 60)
        report.append("VIETNAMESE SLU AUGMENTATION QUALITY REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Samples: {stats['total_samples']:,}")
        report.append(f"Passed All Checks: {stats['passed_all_checks']:,} ({stats['quality_rate']:.1%})")
        report.append(f"Contains Vietnamese: {stats['samples_with_vietnamese']:,} ({stats['vietnamese_rate']:.1%})")
        
        report.append("\n## Failed Checks Distribution:")
        for check, count in stats['failed_checks'].most_common():
            report.append(f"   {check}: {count:,}")
        
        report.append("\n## Intent Distribution:")
        for intent, count in stats['intent_distribution'].most_common(10):
            report.append(f"   {intent}: {count:,}")
        
        report.append("\n## Entity Type Distribution:")
        for entity_type, count in stats['entity_distribution'].most_common(10):
            report.append(f"   {entity_type}: {count:,}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
                
            # Also save detailed stats as JSON
            json_path = output_path.replace('.txt', '_stats.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        return report_text
    
    def filter_quality_samples(self, samples: List[Dict], min_quality: float = 3.0, 
                             min_naturalness: float = 3.0) -> List[Dict]:
        """
        Filter samples based on quality scores
        """
        filtered = []
        
        for sample in samples:
            try:
                quality_score = float(sample.get('Quality_Score_1_5', 0) or 0)
                naturalness_score = float(sample.get('Naturalness_Score_1_5', 0) or 0)
                
                if quality_score >= min_quality and naturalness_score >= min_naturalness:
                    filtered.append(sample)
            except ValueError:
                # Skip samples with invalid scores
                continue
        
        return filtered