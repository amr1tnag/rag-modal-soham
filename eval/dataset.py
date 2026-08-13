"""The golden set: questions with known-correct answers.

Each case names the fact the pipeline has to find, so a failure points at a
specific retrieval or grounding problem rather than a vague low score.

`expected_output` is what a correct answer must convey — not a string to match
exactly. DeepEval's judge compares meaning, so phrasing may differ.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Golden:
    question: str
    expected_output: str
    # Distinctive text from the chunk that ought to be retrieved. Used for the
    # retrieval hit-rate check, which needs no LLM judge to compute.
    must_retrieve: str
    tags: tuple[str, ...] = field(default_factory=tuple)


GOLDENS: list[Golden] = [
    Golden(
        question="Who founded Acme Robotics and in what year?",
        expected_output="Maria Chen and Dev Patel founded it in 1987.",
        must_retrieve="founded in 1987 by Maria Chen and Dev Patel",
        tags=("simple-fact",),
    ),
    Golden(
        question="Where is the head office now, and where was it before?",
        expected_output=(
            "Headquarters are in Lisbon, moved there in 2019 from Rotterdam."
        ),
        must_retrieve="moved from Rotterdam to Lisbon in 2019",
        tags=("simple-fact", "two-part"),
    ),
    Golden(
        question="What were the core hours?",
        expected_output="Core hours are 10:00 to 15:00.",
        must_retrieve="Core hours",
        tags=("simple-fact",),
    ),
    Golden(
        question="How many days a week can I work from home?",
        expected_output="Up to three days per week.",
        must_retrieve="remotely up to three days per week",
        # Phrased in the user's words ("work from home"), not the document's
        # ("work remotely") — this is what semantic retrieval should handle.
        tags=("paraphrased",),
    ),
    Golden(
        question="How much annual leave do I get, and can I carry it over?",
        expected_output=(
            "25 days of paid annual leave. Leave does not carry over, except "
            "up to 5 days into January, which must be used before 31 March."
        ),
        must_retrieve="25 days of paid annual leave",
        tags=("multi-part",),
    ),
    Golden(
        question="I flew to Berlin, a 900 km trip. Did I need prior approval?",
        expected_output=(
            "No. Prior approval is required for air travel under 600 km; at "
            "900 km the rail-preference rule does not apply."
        ),
        must_retrieve="600 kilometres",
        # Requires applying a rule to a number, not just quoting a sentence.
        tags=("reasoning",),
    ),
    Golden(
        question="How long do I have to submit an expense claim?",
        expected_output="Within 60 days of incurring the expense.",
        must_retrieve="within 60 days",
        tags=("simple-fact",),
    ),
    Golden(
        question="How often are passwords rotated?",
        expected_output=(
            "They are not rotated on a schedule; rotation is only required "
            "after a suspected compromise."
        ),
        must_retrieve="never rotated on a schedule",
        # The intuitive answer ("every 90 days") is wrong — this catches a
        # model falling back on training data instead of reading the context.
        tags=("counterintuitive",),
    ),
    Golden(
        question="What is the hotel spending cap outside Lisbon?",
        expected_output="220 euros per night elsewhere in Europe.",
        must_retrieve="220 euros per night",
        # Two similar numbers sit in one sentence; picking the wrong one is a
        # realistic failure.
        tags=("distractor",),
    ),
    Golden(
        question="What is the company's dental insurance policy?",
        expected_output=(
            "The handbook does not cover dental insurance, so the answer is "
            "not available in the provided documents."
        ),
        must_retrieve="",  # nothing should match; refusal is the correct answer
        tags=("unanswerable",),
    ),
]
