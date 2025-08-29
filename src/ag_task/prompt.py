

PROMPT1 = """
Answer the following question concisely in one sentence:

question:
{question}
"""

PROMPT2 = """
Answer the following question concisely in one sentence, you should follow the points:

1. You should answer the question as simple as possible, some questions may just need a word or two.
2. You don't need to answer the question in a very detailed way, just give a concise answer.


question:
{question}
"""

PROMPT3 = """
Answer the following question concisely in one sentence, you should follow the points:

1. You should answer the question as simple as possible, some questions may just need a word or two.
2. You don't need to answer the question in a very detailed way, just give a concise answer.
3. For the answer in number, you should answer the number(one, two, three, etc.) but not 1,2,3, etc. in the question.

question:
{question}
"""

PROMPT4 = """
Answer the following question concisely in one sentence, you should follow the points:

1. You should answer the question as simple as possible, some questions may just need a word or two.
2. You don't need to answer the question in a very detailed way, just give a concise answer.
3. For the answer in number, you should answer the number(one, two, three, etc.) but not 1,2,3, etc. in the question.
4. You need to think about the question and answer it carefully.

question:
{question}
"""

PROMPT5 = """
Answer the following question concisely in one sentence, you should follow the points:

1. You should answer the question as simple as possible, some questions may just need a word or two.
2. You don't need to answer the question in a very detailed way, just give a concise answer.
3. For the answer in number, you should answer the number(one, two, three, etc.) but not 1,2,3, etc. in the question.

the transcript of the video is:
{transcript}

question:
{question}
"""

PROMPT6 = """
Answer the following question concisely in one sentence, you should follow the points:

1. You should answer the question as simple as possible, some questions may just need a word or two.
2. You don't need to answer the question in a very detailed way, just give a concise answer.
3. For the answer in number, you should answer the number(one, two, three, etc.) but not 1,2,3, etc. in the question.
4. You need to think about the question and answer it carefully.
5. If the question is about the audio, you should answer the question based on the transcript of the video.
6. The transcript can be in different languages, you should answer the question in English.


the transcript of the video is:
{transcript}

question:
{question}
"""

PROMPT7 = """
Answer the following question concisely in one sentence, you should follow the points:

1. You should answer the question as simple as possible, some questions may just need a word or two.
2. You don't need to answer the question in a very detailed way, just give a concise answer.
3. For the answer in number, you should answer the number(one, two, three, etc.) but not 1,2,3, etc. in the question.
4. The asr transcript can be in different languages, you should answer the question in English.
5. Base your answer on both the ASR transcript and the video frames; if needed, use reasoning or general knowledge to give the most reasonable answer.



the ASR transcript of the video is:
{transcript}

the question is:
{question}
"""

PROMPT_LIST = [
    PROMPT1,
    PROMPT2,
    PROMPT3,
    PROMPT4,
    PROMPT5,
    PROMPT6,
    PROMPT7
]