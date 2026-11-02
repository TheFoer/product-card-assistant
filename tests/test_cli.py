import json

from stub_llm import StubLlm, fail, ok

from cardgen.cli import main


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def env(monkeypatch, url, key="k"):
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")


def test_generate_to_dir(tmp_path, monkeypatch, dm02_input, dm02_response):
    source = write(tmp_path / "dm.json", dm02_input)
    with StubLlm([ok(dm02_response)]) as stub:
        env(monkeypatch, stub.url)
        code = main(["generate", source, "--out-dir", str(tmp_path / "out")])
    assert code == 0
    assert (tmp_path / "out/dm.md").read_text(encoding="utf-8").startswith("# Манометр")
    assert json.loads((tmp_path / "out/dm.json").read_text(encoding="utf-8"))["status"] == "ready"


def test_batch_continues_after_failure(tmp_path, monkeypatch, capsys, dm02_input, dm02_response):
    source = write(tmp_path / "many.json", [dm02_input, {"name": ""}, dm02_input])
    with StubLlm([ok(dm02_response)]) as stub:
        env(monkeypatch, stub.url)
        code = main(["generate", source, "--out-dir", str(tmp_path / "out")])
    assert code == 1
    assert (tmp_path / "out/many-1.md").exists() and (tmp_path / "out/many-3.md").exists()
    assert not (tmp_path / "out/many-2.md").exists()
    assert "name" in capsys.readouterr().err


def test_generate_llm_failure_exit_code(tmp_path, monkeypatch, dm02_input):
    source = write(tmp_path / "dm.json", dm02_input)
    with StubLlm([fail(401)]) as stub:
        env(monkeypatch, stub.url)
        assert main(["generate", source]) == 1


def test_no_key(tmp_path, monkeypatch, capsys, dm02_input):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["generate", write(tmp_path / "dm.json", dm02_input)]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_check_codes(tmp_path, capsys, dm02_input):
    assert main(["check", write(tmp_path / "ok.json", dm02_input)]) == 0
    assert "Питание" in capsys.readouterr().out
    broken = tmp_path / "bad.json"
    broken.write_text("{", encoding="utf-8")
    assert main(["check", str(broken)]) == 2
    assert main(["check", str(tmp_path / "absent.json")]) == 2
