from src.json_parser import load_prompts, load_functions


def test_valid_prompts(tmp_path):
    file = tmp_path / "prompts.json"
    file.write_text('[{"prompt": "Hello"}]')
    prompts = load_prompts(str(file))
    assert len(prompts) == 1


def test_empty_prompts(tmp_path):
