# PROMPT Evaluatation

## Raw scores

| PROMPT | ROUGE_L | METEOR | BERTScore_F1 | STS_Cosine |
|--------|---------|--------|--------------|------------|
| PROMPT1 | 0.2243 | 0.2227 | 0.2515 | 0.4691 |
| PROMPT2 | 0.3342 | 0.2164 | 0.4762 | 0.6079 |
| PROMPT3 | 0.3894 | 0.2281 | 0.4970 | 0.6314 |
| PROMPT4 | 0.3837 | 0.2278 | 0.4849 | 0.6235 |
| PROMPT5 | 0.4070 | 0.2518 | 0.5081 | 0.6380 |
| PROMPT6 | 0.3905 | 0.2461 | 0.4849 | 0.6209 |
| PROMPT7 | 0.3521 | 0.2456 | 0.4416 | 0.5952 |
| PROMPT5_lora | 0.4014 | 0.2402 | 0.4963 | 0.6298 |
| PROMPT6_lora | 0.3797 | 0.2406 | 0.4922 | 0.6220 |


## Normalized scores

| PROMPT | ROUGE_L | METEOR | BERTScore_F1 | STS_Cosine |
|--------|---------|--------|--------------|------------|
| PROMPT1 | 0.2243 | 0.2720 | 0.1854 | 0.4710 |
| PROMPT2 | 0.3342 | 0.2820 | 0.4369 | 0.6104 |
| PROMPT3 | 0.3894 | 0.2816 | 0.4497 | 0.6312 |
| PROMPT4 | 0.3837 | 0.2829 | 0.4371 | 0.6243 |
| PROMPT5 | 0.4070 | 0.3083 | 0.4656 | 0.6388 |
| PROMPT6 | 0.3905 | 0.3055 | 0.4451 | 0.6246 |
| PROMPT7 | 0.3521 | 0.3096 | 0.3958 | 0.5985 |
| PROMPT5_lora | 0.4014 | 0.2967 | 0.4548 | 0.6324 |
| PROMPT6_lora | 0.3797 | 0.2979 | 0.4475 | 0.6242 |





# analyse

## Some videos may need audio to infer the result

- RjEjcM_Z0oU: "question": "What is heard in the background while the paratha is being brought into the room by the waiter?"

- QN7iuxaKVOE: "question": "What are the lyrics of the song after \"She brung and put a ring on my finger.\""

## Some videos may need reasoning ability to get the right answer

- QhczbmyAPQw: "question": "how many red round patches are on the woman's jeans?"

- RZrkA52TGBA: "question": "How many times balls appear in the video?" (this is a little tricky one)


## Some videos may need more knowledge to answer the question

- PKgoIkEP_6c: "question": "What color is the flower?"

## Some format problem in the answer

- PAmo_TwQvOI: 'generation': One, 'GT': one
