# Vietnamese SLU Data Augmentation - User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Format](#data-format)
4. [Configuration](#configuration)
5. [Augmentation Process](#augmentation-process)
6. [Review Process](#review-process)
7. [Integration](#integration)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Clone and Install
```bash
git clone https://github.com/your-username/vietnamese-slu-augmentation.git
cd vietnamese-slu-augmentation
pip install -r requirements.txt
```

### Option 2: Install from PyPI (if published)
```bash
pip install vietnamese-slu-augmentation
```

### Requirements
- Python 3.7+
- pandas, numpy, pyyaml, tqdm
- Modern web browser (for review dashboard)

## Quick Start

### 1. Prepare Your Data
Create a JSONL file with your SLU data:
```json
{"sentence": "bật đèn phòng khách", "intent": "bật thiết bị", "entities": [{"type": "device", "filler": "đèn"}, {"type": "living_space", "filler": "phòng khách"}]}
```

### 2. Generate Augmentations
```bash
python scripts/1_generate_augmentations.py --input data/input/your_dataset.jsonl
```

### 3. Review Generated Samples
Open `dashboard/index.html` in your browser and load the generated CSV file.

### 4. Integrate Approved Data
```bash
python scripts/3_integrate_approved_data.py --review-file data/output/review/reviewed_samples.csv
```

### 5. Use in Training
Upload the final dataset to your training environment (Kaggle, Colab, etc.).

## Data Format

### Input Format (JSONL)
Each line should contain a JSON object with:
- `sentence`: The Vietnamese text
- `intent`: The intent label
- `entities`: Array of entity objects with `type` and `filler`

```json
{
  "sentence": "bật đèn phòng khách",
  "intent": "bật thiết bị",
  "entities": [
    {"type": "device", "filler": "đèn"},
    {"type": "living_space", "filler": "phòng khách"}
  ]
}
```

### Output Format
The system produces:
1. **Augmented Data JSON**: Generated samples with metadata
2. **Review CSV**: Human-reviewable format
3. **Final Dataset JSONL**: Approved samples ready for training

## Configuration

Edit `config.yaml` to customize augmentation behavior:

### Key Settings
```yaml
augmentation:
  target_samples_per_rare_entity: 100  # Target size for rare categories
  min_entity_frequency: 50             # Entities with <50 examples are rare
  max_variations_per_sample: 3         # Max augmentations per sample

quality_control:
  min_length_ratio: 0.7               # Minimum length (70% of original)
  max_length_ratio: 2.0               # Maximum length (200% of original)
  require_human_review: true          # Require manual approval

vietnamese:
  politeness_level: "mixed"           # formal, informal, or mixed
  enable_regional_dialects: false     # Use regional variations
```

### Vocabulary Customization
Add your domain-specific synonyms:
```yaml
vocabularies:
  devices:
    "smart_tv": ["TV thông minh", "tivi thông minh", "smart TV"]
    "robot_vacuum": ["robot hút bụi", "máy hút bụi tự động"]
```

## Augmentation Process

### 1. Dataset Analysis
The system automatically:
- Counts intent and entity frequencies
- Identifies rare categories needing augmentation
- Reports distribution statistics

### 2. Sample Selection
Samples are selected for augmentation if they contain:
- Rare intents (< `min_intent_frequency` examples)
- Rare entities (< `min_entity_frequency` examples)

### 3. Augmentation Methods

#### Synonym Replacement
- Replaces devices with Vietnamese synonyms
- Preserves entity boundaries and types
- Example: "máy lạnh" → "điều hòa"

#### Politeness Addition
- Adds Vietnamese politeness markers
- Example: "bật đèn" → "Làm ơn bật đèn nhé"

#### Sentence Restructuring
- Moves time/location expressions
- Example: "bật đèn lúc 8 giờ" → "Lúc 8 giờ, bật đèn"

#### Time Variation
- Varies time expressions
- Example: "sáng" → "buổi sáng"

### 4. Quality Control
Each generated sample undergoes:
- Length ratio validation
- Entity preservation check
- Vietnamese character validation
- Basic coherence assessment

## Review Process

### Using the Web Dashboard

#### 1. Load Data
- Open `dashboard/index.html`
- Click "Load CSV" and select your review file
- Or drag and drop the CSV file

#### 2. Review Samples
For each sample, evaluate:
- **Quality Score (1-5)**: How well meaning is preserved
- **Naturalness Score (1-5)**: How natural the Vietnamese sounds
- **Comments**: Optional notes
- **Approval**: Yes/No decision

#### 3. Filtering and Navigation
- Filter by augmentation method
- Filter by review status
- Search specific terms
- Track progress with the progress bar

#### 4. Batch Operations
- Auto-approve high-quality samples (≥4 in both scores)
- Export reviewed data
- Save progress automatically

### Review Guidelines

#### ✅ Approve if:
- Sounds natural in Vietnamese
- Preserves original meaning
- Entities are correctly placed
- Grammar is appropriate

#### ❌ Reject if:
- Awkward or unnatural phrasing
- Meaning changed significantly
- Entities are incorrect or missing
- Grammar errors

#### Scoring Guide:
- **5**: Perfect, sounds completely natural
- **4**: Very good, minor issues
- **3**: Acceptable, some awkwardness
- **2**: Poor, significant issues
- **1**: Unusable, completely wrong

## Integration

### Approval Criteria
Samples are approved if:
- Explicitly marked "yes" in Approved_Yes_No, OR
- Both Quality ≥ 3 AND Naturalness ≥ 3 (and not explicitly rejected)

### Integration Strategies

#### Balanced (default)
- Maintains intent distribution balance
- Prevents over-augmentation of any category

#### Selective
- Only augments the rarest categories
- More conservative approach

#### All
- Uses all approved samples
- Maximum augmentation

### Output Files
- `final_dataset_YYYYMMDD_HHMMSS.jsonl`: Ready for training
- `integration_report_YYYYMMDD_HHMMSS.json`: Detailed statistics

## Advanced Usage

### Custom Augmentation Methods
Extend the `VietnameseSLUAugmenter` class:
```python
class CustomAugmenter(VietnameseSLUAugmenter):
    def _apply_custom_method(self, sentence, entities):
        # Your custom logic here
        return augmented_sentence, augmented_entities, metadata
```

### Programmatic Usage
```python
from src.augmenter import VietnameseSLUAugmenter

# Initialize
augmenter = VietnameseSLUAugmenter("config.yaml")

# Generate augmentations
augmentations = augmenter.generate_augmentations(
    sentence="bật đèn phòng khách",
    intent="bật thiết bị",
    entities=[{"type": "device", "filler": "đèn"}]
)
```

### Integration with Training Pipelines

#### For Kaggle/Colab
```python
# After uploading final_dataset.jsonl to Kaggle
import json

with open('/kaggle/input/augmented-data/final_dataset.jsonl', 'r') as f:
    augmented_data = [json.loads(line) for line in f]

# Add to your processor
for item in augmented_data:
    processor.data.append(item)
```

#### For Local Training
```python
from src.integration import load_augmented_dataset

load_augmented_dataset(processor, 'data/output/final/final_dataset.jsonl')
```

## Troubleshooting

### Common Issues

#### "No augmentations generated"
**Cause**: Input data format issues or no rare categories found
**Solution**: 
- Check JSONL format
- Verify rare entity detection thresholds
- Use `--test-mode` for debugging

#### "Low approval rates (<30%)"
**Cause**: Quality thresholds too strict or poor synonyms
**Solution**:
- Lower quality thresholds to 2.5
- Review Vietnamese synonym dictionaries
- Check for domain-specific terms

#### "CSV loading fails"
**Cause**: File encoding or format issues
**Solution**:
- Ensure UTF-8 encoding
- Check column names match exactly
- Verify no extra commas in data

#### "Integration finds no approved samples"
**Cause**: Review columns not filled correctly
**Solution**:
- Fill Quality_Score_1_5 and Naturalness_Score_1_5
- Or set Approved_Yes_No to "yes"/"no"
- Check for typos in approval values

### Performance Tips

#### For Large Datasets
- Use `--max-review-samples 50` for manageable review batches
- Process in chunks of 1000-5000 samples
- Use parallel processing (`parallel_processing: true` in config)

#### Memory Management
- Set `memory.limit_mb` in config
- Clear browser cache if dashboard becomes slow
- Use JSONL format for large datasets

### Quality Improvement

#### Enhance Synonyms
Add domain-specific synonyms to `config.yaml`:
```yaml
vocabularies:
  devices:
    "your_device": ["synonym1", "synonym2"]
```

#### Regional Dialects
Enable regional variations:
```yaml
vietnamese:
  enable_regional_dialects: true
  include_southern_variants: true
```

#### Custom Quality Checks
Implement domain-specific validation:
```python
def custom_quality_check(original, augmented, entities):
    # Your validation logic
    return is_valid
```

## Best Practices

### Data Preparation
1. Clean your input data first
2. Ensure consistent entity annotation
3. Verify intent labels are accurate
4. Start with a sample dataset for testing

### Review Process
1. Have multiple reviewers for consistency
2. Establish clear quality guidelines
3. Review in batches of 20-50 samples
4. Take breaks to maintain focus

### Integration
1. Start with conservative thresholds
2. Monitor training performance
3. Iterate based on results
4. Keep original data for comparison

### Performance Monitoring
1. Track approval rates by method
2. Monitor training improvements
3. Validate on held-out test set
4. Document what works best

---

**Need more help?** Check the [API Reference](api_reference.md) or [Examples](examples.md)!