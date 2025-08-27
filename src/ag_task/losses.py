"""
Custom loss functions for Answer Generation model training.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional
import re


def extract_answers_from_generation(generated_text: str, num_answers: int = 10) -> List[str]:
    """
    Extract up to num_answers from generated text.
    
    Expected format: "1. answer1\n2. answer2\n..." or similar numbered list.
    Falls back to splitting by newlines if numbered format not found.
    
    Args:
        generated_text: The model's generated text
        num_answers: Maximum number of answers to extract
        
    Returns:
        List of extracted answer strings (may be fewer than num_answers)
    """
    # Try to find numbered list pattern first
    pattern = r'^\d+\.?\s*(.+)$'
    lines = generated_text.strip().split('\n')
    
    answers = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try numbered format
        match = re.match(pattern, line)
        if match:
            answers.append(match.group(1).strip())
        else:
            # Fallback: treat each non-empty line as an answer
            if len(answers) < num_answers:
                answers.append(line)
        
        if len(answers) >= num_answers:
            break
    
    # If we still don't have enough answers, pad with the last answer or empty string
    while len(answers) < num_answers:
        if answers:
            answers.append(answers[-1])  # Repeat last answer
        else:
            answers.append("")  # Empty fallback
    
    return answers[:num_answers]


class BERTScoreMaxLoss:
    """
    Loss function that maximizes the best BERTScore between generated answers and ground truth.
    
    The model is trained to generate multiple answers, and we want to maximize:
    max(bertscore(a1, gt), bertscore(a2, gt), ..., bertscore(a10, gt))
    
    This is implemented as: -log(max(bertscore_probs)) where bertscore_probs are softmax-normalized.
    """
    
    def __init__(
        self, 
        num_answers: int = 10,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        device: Optional[torch.device] = None
    ):
        """
        Args:
            num_answers: Number of answers to generate and score
            bertscore_model: Model name for BERTScore computation
            device: Device for BERTScore computation
        """
        self.num_answers = num_answers
        self.bertscore_model = bertscore_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize BERTScore
        try:
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(
                model_type=bertscore_model,
                device=self.device,
                batch_size=64,
                nthreads=4
            )
        except ImportError:
            raise RuntimeError(
                "BERTScore is required. Install with: pip install bert-score"
            )
    
    def compute_bert_scores(self, generated_answers: List[List[str]], ground_truths: List[str]) -> torch.Tensor:
        """
        Compute BERTScore F1 scores for all generated answers against ground truths.
        
        Args:
            generated_answers: List of [answer_list] for each example in batch
            ground_truths: List of ground truth strings
            
        Returns:
            Tensor of shape (batch_size, num_answers) with BERTScore F1 scores
        """
        batch_size = len(generated_answers)
        
        # Flatten all candidate answers and corresponding references
        all_candidates = []
        all_references = []
        
        for i, (answers, gt) in enumerate(zip(generated_answers, ground_truths)):
            for answer in answers:
                all_candidates.append(answer)
                all_references.append(gt)
        
        # Compute BERTScores
        if all_candidates:
            _, _, f1_scores = self.bert_scorer.score(all_candidates, all_references)
            f1_scores = f1_scores.cpu()
        else:
            f1_scores = torch.zeros(len(all_candidates))
        
        # Reshape back to (batch_size, num_answers)
        scores = f1_scores.view(batch_size, self.num_answers)
        return scores.to(self.device)
    
    def __call__(
        self, 
        generated_texts: List[str], 
        ground_truths: List[str],
        return_scores: bool = False
    ) -> torch.Tensor:
        """
        Compute the BERTScore max loss.
        
        Args:
            generated_texts: List of generated text strings from the model
            ground_truths: List of ground truth answer strings
            return_scores: If True, also return the BERTScores
            
        Returns:
            Loss tensor (scalar)
            If return_scores=True, returns (loss, bert_scores)
        """
        # Extract answers from generated texts
        generated_answers = []
        for text in generated_texts:
            answers = extract_answers_from_generation(text, self.num_answers)
            generated_answers.append(answers)
        
        # Compute BERTScores
        bert_scores = self.compute_bert_scores(generated_answers, ground_truths)
        
        # Apply softmax to convert scores to probabilities
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        score_probs = F.softmax(bert_scores, dim=1) + epsilon
        
        # Compute max probability for each example
        max_probs, _ = torch.max(score_probs, dim=1)
        
        # Loss is negative log of max probability
        loss = -torch.log(max_probs).mean()
        
        if return_scores:
            return loss, bert_scores
        return loss


class SimpleBERTScoreMaxLoss:
    """
    Simplified version that directly maximizes the max BERTScore without softmax transformation.
    """
    
    def __init__(
        self, 
        num_answers: int = 10,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        device: Optional[torch.device] = None
    ):
        self.bert_score_loss = BERTScoreMaxLoss(num_answers, bertscore_model, device)
    
    def __call__(
        self, 
        generated_texts: List[str], 
        ground_truths: List[str],
        return_scores: bool = False
    ) -> torch.Tensor:
        """
        Compute loss as: -mean(max(bert_scores, dim=1))
        
        This directly maximizes the best BERTScore for each example.
        """
        # Extract answers from generated texts
        generated_answers = []
        for text in generated_texts:
            answers = extract_answers_from_generation(text, self.bert_score_loss.num_answers)
            generated_answers.append(answers)
        
        # Compute BERTScores
        bert_scores = self.bert_score_loss.compute_bert_scores(generated_answers, ground_truths)
        
        # Take max score for each example and negate for loss
        max_scores, _ = torch.max(bert_scores, dim=1)
        loss = -max_scores.mean()
        
        if return_scores:
            return loss, bert_scores
        return loss
