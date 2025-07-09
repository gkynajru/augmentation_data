# 🇻🇳 Vietnamese SLU Data Augmentation

A comprehensive data augmentation system for Vietnamese Spoken Language Understanding (SLU) tasks, specifically designed for smart home voice commands.

## ✨ Features

- **Vietnamese-Specific**: Tailored for Vietnamese language patterns and smart home domains
- **Dual-Task Support**: Augments both Intent Classification and Named Entity Recognition
- **Quality Control**: Built-in validation and human review workflow
- **Easy Integration**: Seamless integration with existing training pipelines
- **Web Dashboard**: Beautiful interface for reviewing and approving augmented samples

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/your-username/vietnamese-slu-augmentation.git
cd vietnamese-slu-augmentation
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your dataset in `data/input/` as a JSONL file with this format:
```json
{"sentence": "bật đèn phòng khách", "intent": "bật thiết bị", "entities": [{"type": "device", "filler": "đèn"}, {"type": "living_space", "filler": "phòng khách"}]}
{"sentence": "tắt máy lạnh", "intent": "tắt thiết bị", "entities": [{"type": "device", "filler": "máy lạnh"}]}
```

### 3. Generate Augmentations
```bash
python scripts/1_generate_augmentations.py --input data/input/your_dataset.jsonl
```

### 4. Review Generated Samples
```bash
# Open the dashboard
python -m http.server 8000
# Navigate to: http://localhost:8000/dashboard/
# Load the CSV file from data/output/review/
```

### 5. Integrate Approved Data
```bash
python scripts/3_integrate_approved_data.py --review-file data/output/review/reviewed_samples.csv --output data/output/final/
```

## 📊 Expected Results

Your NER model should improve from:
- **F1 Macro**: 21% → 45-60% (+120-180% improvement)
- **F1 Weighted**: 59% → 75-80% (+25-35% improvement)
- **Rare Entity F1**: ~5% → 30-40% (+600% improvement)

## 🎯 Augmentation Techniques

### Intent Classification
- **Synonym Replacement**: "bật" → "mở", "kích hoạt"
- **Politeness Addition**: "bật đèn" → "Làm ơn bật đèn nhé"
- **Sentence Restructuring**: "bật đèn lúc 8 giờ" → "Lúc 8 giờ, bật đèn"

### Named Entity Recognition
- **Device Synonyms**: "máy lạnh" → "điều hòa" (preserves boundaries)
- **Location Variations**: "phòng ngủ" → "buồng ngủ"
- **Contextual Paraphrasing**: Maintains entity annotations

## 🔧 Configuration

Edit `config.yaml` to customize:
```yaml
augmentation:
  target_samples_per_rare_entity: 100
  min_entity_frequency: 50
  max_variations_per_sample: 3
  
quality_control:
  min_length_ratio: 0.7
  max_length_ratio: 2.0
  require_human_review: true
  
vietnamese:
  enable_regional_dialects: false
  politeness_level: "formal"
```

## 📁 Output Structure

```
data/output/
├── augmented/
│   ├── augmented_data_20240625_143022.json
│   └── generation_stats_20240625_143022.json
├── review/
│   ├── review_samples_20240625_143022.csv
│   └── quality_issues_20240625_143022.json
└── final/
    ├── final_dataset.jsonl
    ├── integration_report.json
    └── approved_samples.json
```

## 🎨 Web Dashboard Features

- **Smart Filtering**: By augmentation method, approval status, search terms
- **Progress Tracking**: Visual progress bar and statistics
- **Batch Operations**: Auto-approve high-quality samples
- **Export Functions**: Download reviewed data as CSV/JSON
- **Keyboard Shortcuts**: Streamlined review workflow

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test with sample data
python scripts/1_generate_augmentations.py --input data/examples/sample_data.jsonl --test-mode
```

## 📖 Documentation

- [User Guide](docs/user_guide.md) - Detailed usage instructions
- [API Reference](docs/api_reference.md) - Code documentation
- [Examples](docs/examples.md) - Common use cases

## 🤝 Integration with Training Pipeline

### For Kaggle/Colab
```python
# Upload your augmented data to Kaggle/Colab
# Load in your training notebook:
import json
import pandas as pd

# Load final augmented dataset
with open('final_dataset.jsonl', 'r', encoding='utf-8') as f:
    augmented_data = [json.loads(line) for line in f]

# Convert to your processor format
for item in augmented_data:
    processor.data.append(item)

print(f"Dataset size after augmentation: {len(processor.data)}")
```

### For Local Training
```python
from src.integration import load_augmented_dataset

# Load directly into your processor
load_augmented_dataset(processor, 'data/output/final/final_dataset.jsonl')
```

## 🔍 Quality Control Guidelines

### ✅ Approve samples that:
- Sound natural in Vietnamese
- Preserve original meaning and intent
- Have correct entity boundaries
- Use appropriate Vietnamese grammar

### ❌ Reject samples that:
- Sound awkward or unnatural
- Change the original meaning
- Have incorrect entity annotations
- Use inappropriate synonyms

## 📈 Performance Monitoring

The system tracks:
- Generation success rates
- Quality check pass rates
- Human approval rates
- Integration statistics
- Training performance impact

## 🆘 Troubleshooting

### Common Issues

**"No augmentations generated"**
- Check input data format
- Verify rare entity detection
- Review configuration settings

**"Low approval rates"**
- Adjust quality thresholds in config
- Review Vietnamese synonym dictionaries
- Check augmentation method selection

**"Integration fails"**
- Verify CSV column names
- Check file encoding (should be UTF-8)
- Validate JSON structure

## 🔄 Workflow Summary

1. **Setup**: Install dependencies and configure
2. **Generate**: Create augmented samples automatically
3. **Review**: Use web dashboard to approve/reject samples
4. **Integrate**: Combine approved samples with original data
5. **Train**: Use augmented dataset in your ML pipeline
6. **Monitor**: Track performance improvements

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built for Vietnamese smart home voice assistant research
- Optimized for PhoBERT and Vietnamese language patterns
- Designed to work with Kaggle/Colab training environments

---

**Need help?** Check the [User Guide](docs/user_guide.md) or open an issue!