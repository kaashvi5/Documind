import os
from types import SimpleNamespace

import pytest

from rag.generator import ExtractiveGenerator, build_prompt, find_supporting_chunks, is_not_found


def count(text):
    return len(text.split())


def truncate(text, n):
    return " ".join(text.split()[:n])


@pytest.mark.parametrize("text", ["", "unanswerable", "Unanswerable.", "I don't know", "None", "not mentioned"])
def test_not_found_phrases(text):
    assert is_not_found(text)


def test_real_answers_are_not_flagged():
    assert not is_not_found("$500 one time") and not is_not_found("Yes")


def test_prompt_contains_grounding_instruction_and_numbered_context():
    prompt, used, truncated = build_prompt("What is X?", ["alpha text", "beta text"], count, truncate, 500)
    assert "using only the information in the context" in prompt and "unanswerable" in prompt
    assert "[1] alpha text" in prompt and "[2] beta text" in prompt and prompt.rstrip().endswith("sentences:")
    assert used == 2 and not truncated


def test_prompt_respects_token_budget_and_keeps_question():
    texts = ["word " * 200, "second " * 200, "third " * 200]
    prompt, used, truncated = build_prompt("What is the answer to everything?", texts, count, truncate, 300)
    assert count(prompt) <= 300 and truncated and used < 3
    assert "What is the answer to everything?" in prompt and "[1]" in prompt


def test_supporting_chunks_follow_answer_wording():
    texts = ["The stipend is five hundred dollars paid once.", "Volcanoes erupt molten lava.", "Core hours are ten to three."]
    cited, coverage = find_supporting_chunks("A stipend of five hundred dollars.", texts)
    assert cited == [1] and coverage > 0.5
    cited, coverage = find_supporting_chunks("Completely unrelated zebra galaxy", texts)
    assert coverage == 0.0 and cited == [1]
    assert 2 in find_supporting_chunks("See [2] for details", texts)[0]


def chunks(*texts):
    return [SimpleNamespace(text=t) for t in texts]


def test_extractive_generator_is_grounded():
    ctx = chunks("Employees receive a one-time stipend of $500. Core hours are 10:00 to 15:00.", "Volcanoes erupt lava.")
    result = ExtractiveGenerator().generate("How large is the stipend?", ctx)
    assert result.found and "$500" in result.answer and "Volcanoes" not in result.answer


def test_extractive_generator_admits_missing_information():
    assert not ExtractiveGenerator().generate("Who painted the Mona Lisa?", chunks("Employees receive a stipend.")).found
    assert not ExtractiveGenerator().generate("???", chunks("Employees receive a stipend.")).found


@pytest.mark.skipif(os.environ.get("DOCUMIND_RUN_MODEL_TESTS") != "1", reason="set DOCUMIND_RUN_MODEL_TESTS=1 to download and test FLAN-T5")
def test_flan_t5_answers_from_context_and_declines_otherwise():
    from rag.generator import FlanT5Generator

    gen = FlanT5Generator("google/flan-t5-small", device="cpu").load()
    ctx = chunks("Remote employees receive a one-time home-office stipend of $500 and a monthly internet reimbursement of $40.")
    assert "500" in gen.generate("How much is the home-office stipend?", ctx).answer
    assert not gen.generate("What is the capital of France?", chunks("Volcanoes erupt molten lava from the earth.")).found
