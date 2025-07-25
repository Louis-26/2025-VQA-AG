# Answer Generation (AG) Task

## Task Description

The Answer Generation task challenges participants to develop models that can generate relevant textual answers to questions about video content. Given a video and a question, systems must produce up to 10 ranked answers along with generation time metrics.

## Input Format

### Topics File
```
Q_ID, Video_ID, Question
1, tui89Xr_iri, what happened after the woman entered the room?
2, abc123def, how many people are in the scene?
3, xyz789ghi, what is the main activity shown in the video?
```

### Video Format
- Duration: ~30 seconds
- Format: YouTube shorts (accessible via Video_ID)
- Resolution: Variable (standard YouTube quality)
- Test Dataset: ~2000 YouTube shorts (shared with MC task)

## Output Requirements

### Submission Format
```csv
Q_ID, Video_ID, Rank, Answer, Time (sec)
1, tui89Xr_iri, 1, she found a surprise birthday party, 5
1, tui89Xr_iri, 2, she found party, 6
1, tui89Xr_iri, 3, she found a group of people, 8
...
1, tui89Xr_iri, 10, a dog barked at her, 10
```

### Requirements
- Maximum 10 answers per question
- Answers must be in natural language
- Answers should be ranked by confidence/relevance
- Time must be reported in seconds (real-time generation)
- All answers must be unique for each question

## Evaluation Metrics

### Primary Metrics
1. **NDCG (Normalized Discounted Cumulative Gain)**
   - Measures ranking quality
   - Higher weight for top-ranked answers

### Secondary Metrics
1. **STS (Semantic Textual Similarity)**
   - Measures semantic similarity to reference answers
   - Uses sentence transformers

2. **METEOR**
   - Considers synonyms and paraphrases
   - Balanced precision/recall

3. **BERTScore**
   - Contextual embedding similarity
   - Robust to paraphrasing

### Efficiency Metrics
- Average generation time per answer
- Total system runtime
