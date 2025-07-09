#!/usr/bin/env python3
"""
Utility functions for Vietnamese SLU Data Augmentation
======================================================
"""

import json
import csv
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from JSONL file
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    # Validate required fields
                    if 'sentence' not in item:
                        logging.warning(f"Line {line_num}: Missing 'sentence' field")
                        continue
                    
                    if 'intent' not in item:
                        logging.warning(f"Line {line_num}: Missing 'intent' field")
                        continue
                    
                    # Ensure entities field exists
                    if 'entities' not in item:
                        item['entities'] = []
                    
                    data.append(item)
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"Line {line_num}: Invalid JSON - {e}")
                    continue
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {e}")
    
    logging.info(f"Loaded {len(data)} samples from {file_path}")
    return data

def save_jsonl(data: List[Dict], file_path: str) -> None:
    """
    Save data to JSONL file
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logging.info(f"Saved {len(data)} samples to {file_path}")

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    logging.info(f"Saved data to {file_path}")

def validate_dataset_format(data: List[Dict]) -> Dict[str, Any]:
    """
    Validate dataset format and return statistics
    
    Args:
        data: Dataset to validate
        
    Returns:
        Validation report
    """
    report = {
        'total_samples': len(data),
        'valid_samples': 0,
        'issues': [],
        'field_coverage': {
            'sentence': 0,
            'intent': 0,
            'entities': 0
        },
        'entity_stats': {
            'total_entities': 0,
            'unique_entity_types': set(),
            'samples_with_entities': 0
        }
    }
    
    for i, item in enumerate(data):
        sample_valid = True
        
        # Check required fields
        if 'sentence' not in item or not item['sentence'].strip():
            report['issues'].append(f"Sample {i}: Missing or empty 'sentence'")
            sample_valid = False
        else:
            report['field_coverage']['sentence'] += 1
        
        if 'intent' not in item or not item['intent'].strip():
            report['issues'].append(f"Sample {i}: Missing or empty 'intent'")
            sample_valid = False
        else:
            report['field_coverage']['intent'] += 1
        
        if 'entities' not in item:
            report['issues'].append(f"Sample {i}: Missing 'entities' field")
            item['entities'] = []  # Fix it
        else:
            report['field_coverage']['entities'] += 1
            
            # Validate entities
            entities = item['entities']
            if entities:
                report['entity_stats']['samples_with_entities'] += 1
                
            for j, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    report['issues'].append(f"Sample {i}, Entity {j}: Not a dictionary")
                    continue
                
                if 'type' not in entity or not entity['type'].strip():
                    report['issues'].append(f"Sample {i}, Entity {j}: Missing 'type'")
                    continue
                
                if 'filler' not in entity or not entity['filler'].strip():
                    report['issues'].append(f"Sample {i}, Entity {j}: Missing 'filler'")
                    continue
                
                report['entity_stats']['total_entities'] += 1
                report['entity_stats']['unique_entity_types'].add(entity['type'])
        
        if sample_valid:
            report['valid_samples'] += 1
    
    # Convert set to list for JSON serialization
    report['entity_stats']['unique_entity_types'] = list(report['entity_stats']['unique_entity_types'])
    
    return report

def clean_text(text: str) -> str:
    """
    Clean Vietnamese text
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters except Vietnamese diacritics
    # Keep basic punctuation
    text = re.sub(r'[^\w\s\-.,!?áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', text)
    
    return text.strip()

def normalize_entities(entities: List[Dict]) -> List[Dict]:
    """
    Normalize entity format
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        Normalized entities
    """
    normalized = []
    
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        
        normalized_entity = {
            'type': clean_text(entity.get('type', '')),
            'filler': clean_text(entity.get('filler', ''))
        }
        
        # Skip empty entities
        if normalized_entity['type'] and normalized_entity['filler']:
            normalized.append(normalized_entity)
    
    return normalized

def get_entity_statistics(data: List[Dict]) -> Dict[str, Any]:
    """
    Get entity statistics from dataset
    
    Args:
        data: Dataset
        
    Returns:
        Entity statistics
    """
    from collections import Counter
    
    entity_type_counts = Counter()
    entity_filler_counts = Counter()
    samples_with_entities = 0
    total_entities = 0
    
    for item in data:
        entities = item.get('entities', [])
        
        if entities:
            samples_with_entities += 1
        
        for entity in entities:
            if isinstance(entity, dict):
                entity_type = entity.get('type', '').strip()
                entity_filler = entity.get('filler', '').strip()
                
                if entity_type:
                    entity_type_counts[entity_type] += 1
                    total_entities += 1
                
                if entity_filler:
                    entity_filler_counts[entity_filler] += 1
    
    return {
        'total_entities': total_entities,
        'unique_entity_types': len(entity_type_counts),
        'unique_entity_fillers': len(entity_filler_counts),
        'samples_with_entities': samples_with_entities,
        'samples_without_entities': len(data) - samples_with_entities,
        'entity_type_distribution': dict(entity_type_counts.most_common(20)),
        'entity_filler_distribution': dict(entity_filler_counts.most_common(20)),
        'average_entities_per_sample': total_entities / len(data) if data else 0
    }

def create_entity_mapping(data: List[Dict]) -> Dict[str, int]:
    """
    Create entity type to ID mapping
    
    Args:
        data: Dataset
        
    Returns:
        Entity type mapping
    """
    entity_types = set()
    
    for item in data:
        for entity in item.get('entities', []):
            if isinstance(entity, dict) and 'type' in entity:
                entity_types.add(entity['type'])
    
    # Create mapping with 'O' as first label
    mapping = {'O': 0}
    
    for i, entity_type in enumerate(sorted(entity_types), 1):
        mapping[f'B-{entity_type}'] = i * 2 - 1
        mapping[f'I-{entity_type}'] = i * 2
    
    return mapping

def export_for_training(data: List[Dict], output_dir: str, format: str = "jsonl") -> str:
    """
    Export data in training-ready format
    
    Args:
        data: Dataset to export
        output_dir: Output directory
        format: Export format (jsonl, json, csv)
        
    Returns:
        Output file path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = Path().resolve().name
    
    if format == "jsonl":
        output_file = f"{output_dir}/training_data.jsonl"
        save_jsonl(data, output_file)
    
    elif format == "json":
        output_file = f"{output_dir}/training_data.json"
        save_json(data, output_file)
    
    elif format == "csv":
        output_file = f"{output_dir}/training_data.csv"
        
        # Flatten for CSV
        flattened = []
        for item in data:
            flattened.append({
                'sentence': item['sentence'],
                'intent': item['intent'],
                'entities_json': json.dumps(item['entities'], ensure_ascii=False),
                'entity_count': len(item['entities'])
            })
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if flattened:
                writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Exported {len(data)} samples to {output_file}")
    return output_file

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if logging to file
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        filename=log_file,
        filemode='a' if log_file else None
    )
    
    # Also log to console if logging to file
    if log_file:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)

def check_file_encoding(file_path: str) -> str:
    """
    Check file encoding
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected encoding
    """
    import chardet
    
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def convert_to_utf8(file_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert file to UTF-8 encoding
    
    Args:
        file_path: Input file path
        output_path: Output file path (overwrites input if None)
        
    Returns:
        Output file path
    """
    if output_path is None:
        output_path = file_path
    
    # Detect current encoding
    encoding = check_file_encoding(file_path)
    
    if encoding.lower() in ['utf-8', 'ascii']:
        logging.info(f"File {file_path} is already UTF-8")
        return file_path
    
    # Convert to UTF-8
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logging.info(f"Converted {file_path} from {encoding} to UTF-8")
    return output_path