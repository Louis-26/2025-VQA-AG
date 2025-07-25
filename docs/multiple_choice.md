# Multiple Choice (MC) Task

## Task Description

The Multiple Choice task evaluates a model's ability to rank provided answer options for questions about video content. Given a video, a question, and four answer options, systems must sort these options from most to least likely correct.

## Input Format

### Topics File
```
Q_ID, Video_ID, Question, option_1, option_2, option_3, option_4
1, tui89Xr_iri, what happened after the woman entered the room?, found a party, the room was empty, a man surprised her, a dog jumped on her
2, abc123def, what color was the car?, red, blue, green, yellow
3, xyz789ghi, how many times did the bell ring?, once, twice, three times, four times
```

### Video Format
- Duration: ~30 seconds
- Format: YouTube shorts (accessible via Video_ID)
- Resolution: Variable (standard YouTube quality)
- Test Dataset: ~2000 YouTube shorts (shared with AG task)

## Output Requirements

### Submission Format
```csv
Q_ID, Video_ID, Rank, option_X
1, tui89Xr_iri, 1, the room was empty
1, tui89Xr_iri, 2, a dog jumped on her
1, tui89Xr_iri, 3, found a party
1, tui89Xr_iri, 4, a man surprised her
```

### Requirements
- All 4 options must be ranked for each question
- Rank 1 = most likely correct
- Rank 4 = least likely correct
- Each option should appear exactly once per question
- Use exact option text from input file

## Evaluation Metrics

### Primary Metrics
1. **Top-1 Accuracy**
   - Percentage of questions where rank 1 answer is correct
   - Main performance indicator

2. **Mean Reciprocal Rank (MRR)**
   - Average of reciprocal ranks of correct answers
   - Formula: MRR = (1/N) × Σ(1/rank_of_correct_answer)

### Secondary Metrics
- Top-2 Accuracy
- Top-3 Accuracy
- Average rank of correct answer
- Confusion matrix analysis

## Best Practices

### Model Development
- Leverage both visual and textual understanding
- Consider temporal context in videos
- Use attention mechanisms for question-video alignment
- Implement robust option comparison strategies

### Ranking Strategy
- Compare all options simultaneously
- Use relative scoring rather than absolute thresholds
- Consider semantic similarity between options
- Account for negations and subtle differences

### Common Approaches
1. **Cross-modal Matching**: Score each option against video features
2. **Contrastive Learning**: Learn to distinguish correct from incorrect
3. **Multi-choice Transformers**: Specialized architectures for MC tasks
4. **Ensemble Methods**: Combine multiple ranking strategies
