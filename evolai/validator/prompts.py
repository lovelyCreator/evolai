"""
Judge Prompt Templates for EvolAI Evaluation System

All judge prompts are centralized here. Miner content is ALWAYS wrapped in
XML delimiters (<miner_answer> / <candidate_response>) and placed in the
user section — never in the system prompt.

Design basis: VALIDATOR_INTERVIEW_EVALUATION.md §Phase 1-3 prompts.
"""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Validity Check
# ──────────────────────────────────────────────────────────────────────────────

VALIDITY_SYSTEM = (
    "You are an evaluation assistant. You evaluate whether a model's "
    "response is valid for further testing. Respond ONLY with JSON."
)

VALIDITY_USER = """\
Instruction: {instruction}
Model Response:
<miner_answer>
{sanitized_response}
</miner_answer>

Evaluate the response based on the following criteria:
1. Is it in natural language (understandable by humans)?
2. Is the content related to the instruction?
3. Does it demonstrate an attempt to address the topic?

Note: The response does NOT need to be correct or complete.

Return your evaluation in JSON format:
{{
  "valid": "yes" or "no",
  "correctness": <float between 0.0 and 1.0>,
  "reasoning": "brief explanation"
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Knowledge Scoring
# ──────────────────────────────────────────────────────────────────────────────

SCORING_SYSTEM = (
    "You are a teacher evaluating a student's answer to a question. "
    "Respond ONLY with JSON."
)

SCORING_USER = """\
Question: {question}
Correct Answer: {reference_answer}
<miner_answer>
{sanitized_response}
</miner_answer>

Evaluate the student's response and assign a score from 0.0 to 1.0:

Scoring Guidelines:
- 0.0: Completely wrong, no understanding demonstrated
- 0.1-0.3: Shows minimal understanding, major errors
- 0.4-0.6: Partially correct, some key concepts present
- 0.7-0.9: Mostly correct, minor issues or incomplete
- 1.0: Fully correct and complete

Return your evaluation in JSON format:
{{
  "score": <float between 0.0 and 1.0>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "detailed explanation",
  "strengths": ["list of correct points"],
  "weaknesses": ["list of errors or gaps"]
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Interview — Initial Analysis
# ──────────────────────────────────────────────────────────────────────────────

INTERVIEW_INITIAL_SYSTEM = (
    "You are conducting a technical interview to assess deep understanding. "
    "Respond ONLY with JSON."
)

INTERVIEW_INITIAL_USER = """\
Context:
- Original Question: {question}
- Reference Answer: {reference_answer}
<candidate_response>
{sanitized_response}
</candidate_response>

Tasks:
1. Summarize the candidate's answer (strengths and weaknesses)
2. Identify knowledge gaps or unclear reasoning
3. Generate ONE challenging follow-up question to probe understanding

The follow-up question should:
- Target a specific weakness or gap
- Require deeper explanation, not re-statement
- Test genuine understanding (not memorization)

Return in JSON format:
{{
  "summary": "brief summary of candidate's answer",
  "weaknesses": ["identified gaps or issues"],
  "follow_up_question": "the question to ask next"
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Interview — Follow-up Generation
# ──────────────────────────────────────────────────────────────────────────────

INTERVIEW_FOLLOWUP_SYSTEM = (
    "You are a rigorous technical interviewer probing whether a candidate truly "
    "understands a topic or is just pattern-matching. "
    "Respond ONLY with JSON."
)

INTERVIEW_FOLLOWUP_USER = """\
Conversation so far (turn {turn_num} of {total_turns}):
{history_text}

Previous Analysis Summary:
{previous_summary}

You must ask another probing question unless turn_num >= total_turns.
Focus on:
- Gaps, vague claims, or memorised phrases in the candidate's last answer
- Edge cases, failure modes, or "why" behind what they said
- Cross-concept connections they haven't demonstrated yet
- Correcting or challenging anything that was wrong or unclear

Only set continue_interview=false if turn_num >= total_turns or the candidate
has exhaustively demonstrated deep mastery from every angle.

Return JSON:
{{
  "analysis": "one-sentence assessment of the latest answer",
  "continue_interview": true or false,
  "next_question": "the next challenging follow-up question, or null if done",
  "reasoning": "why this question or why stopping"
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Interview — Final Verdict
# ──────────────────────────────────────────────────────────────────────────────

INTERVIEW_VERDICT_SYSTEM = (
    "You are a technical interviewer giving a final assessment. "
    "Respond ONLY with JSON."
)

INTERVIEW_VERDICT_USER = """\
Original Question: {question}
Reference Answer: {reference_answer}

Interview Summary (accumulated analysis across all turns):
{interview_summary}

Based on the COMPLETE interview, rate the candidate's genuine understanding.

Consider:
- Did they explain reasoning or just recite memorized facts?
- Did they handle follow-up challenges and edge cases?
- Were answers consistent, or did they contradict themselves?
- Did they demonstrate ability to think through new variations?

A high score (0.8-1.0) requires demonstrated reasoning under pressure, not just
a correct initial answer.

Return JSON:
{{
  "interview_score": <float 0.0-1.0>,
  "genuine_understanding": "yes" or "partial" or "no",
  "reasoning": "one sentence explanation"
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# Compaction: Summarize older interview turns
# ──────────────────────────────────────────────────────────────────────────────

COMPACTION_SYSTEM = (
    "You are a concise note-taker. Summarize interview exchanges preserving "
    "key facts. Return plain text only."
)

COMPACTION_USER = """\
Summarize the following interview exchange concisely.
Preserve: candidate strengths, weaknesses, identified knowledge gaps,
which topics were explored, and notable inconsistencies.

{history_text}

Return plain text summary, max 600 words."""


# ──────────────────────────────────────────────────────────────────────────────
# Builder helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_validity_messages(
    instruction: str,
    sanitized_response: str,
) -> list[dict]:
    """Build chat messages for the validity check judge call."""
    return [
        {"role": "system", "content": VALIDITY_SYSTEM},
        {"role": "user", "content": VALIDITY_USER.format(
            instruction=instruction,
            sanitized_response=sanitized_response,
        )},
    ]


def build_scoring_messages(
    question: str,
    reference_answer: str,
    sanitized_response: str,
) -> list[dict]:
    """Build chat messages for the knowledge scoring judge call."""
    return [
        {"role": "system", "content": SCORING_SYSTEM},
        {"role": "user", "content": SCORING_USER.format(
            question=question,
            reference_answer=reference_answer,
            sanitized_response=sanitized_response,
        )},
    ]


def build_initial_interview_messages(
    question: str,
    reference_answer: str,
    sanitized_response: str,
) -> list[dict]:
    """Build chat messages for the initial interview analysis."""
    return [
        {"role": "system", "content": INTERVIEW_INITIAL_SYSTEM},
        {"role": "user", "content": INTERVIEW_INITIAL_USER.format(
            question=question,
            reference_answer=reference_answer,
            sanitized_response=sanitized_response,
        )},
    ]


def build_followup_interview_messages(
    history_text: str,
    previous_summary: str,
    turn_num: int = 1,
    total_turns: int = 10,
) -> list[dict]:
    """Build chat messages for follow-up interview question generation."""
    return [
        {"role": "system", "content": INTERVIEW_FOLLOWUP_SYSTEM},
        {"role": "user", "content": INTERVIEW_FOLLOWUP_USER.format(
            history_text=history_text,
            previous_summary=previous_summary,
            turn_num=turn_num,
            total_turns=total_turns,
        )},
    ]


def build_final_interview_verdict_messages(
    question: str,
    reference_answer: str,
    interview_summary: str,
) -> list[dict]:
    """Build chat messages for the final per-question interview verdict."""
    return [
        {"role": "system", "content": INTERVIEW_VERDICT_SYSTEM},
        {"role": "user", "content": INTERVIEW_VERDICT_USER.format(
            question=question,
            reference_answer=reference_answer,
            interview_summary=interview_summary,
        )},
    ]


def build_compaction_messages(history_text: str) -> list[dict]:
    """Build chat messages for context compaction."""
    return [
        {"role": "system", "content": COMPACTION_SYSTEM},
        {"role": "user", "content": COMPACTION_USER.format(
            history_text=history_text,
        )},
    ]
