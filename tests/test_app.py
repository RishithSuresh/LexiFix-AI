import json
import urllib.request
import urllib.error
import pytest
from app import start_server, find_free_port


@pytest.fixture(scope="module")
def server_url():
    port = find_free_port(8890)
    server = start_server("127.0.0.1", port)
    url = f"http://127.0.0.1:{port}"
    yield url
    server.shutdown()


def test_static_index_serving(server_url):
    req = urllib.request.Request(f"{server_url}/")
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        content = response.read().decode("utf-8")
        assert "LexiFix AI" in content
        assert "Spelling Intelligence Studio" in content


def test_api_info_endpoint(server_url):
    req = urllib.request.Request(f"{server_url}/api/info")
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "torch_device" in data
        assert "python_version" in data


def test_api_vocabulary_endpoint(server_url):
    req = urllib.request.Request(f"{server_url}/api/vocabulary")
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["size"] > 0
        assert data["tokens"] > 0
        assert len(data["top"]) > 0


def test_api_explain_word(server_url):
    payload = json.dumps({"word": "recieve"}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/explain/word",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["word"] == "recieve"
        assert data["in_vocabulary"] is False
        assert data["corrected"] == "receive"
        assert data["distance"] == 2
        assert "soundex" in data
        assert "damerau_distance" in data
        assert len(data["candidates"]) > 0


def test_api_correct_baseline(server_url):
    payload = json.dumps({
        "text": "I recieve the mesage from my frend.",
        "max_distance": 2,
        "models": [],
        "expected_text": "I receive the message from my friend."
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{server_url}/api/correct",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["baseline_text"] == "I receive the message from my friend."
        assert len(data["corrections"]) == 3
        assert data["baseline_accuracy"] == 1.0
        assert data["baseline_f1"] == 1.0
        assert "baseline_cer" in data
        assert "baseline_wer" in data


def test_api_csv_list_and_read(server_url):
    req = urllib.request.Request(f"{server_url}/api/csv/list")
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        files = json.loads(response.read().decode("utf-8"))
        assert isinstance(files, list)
        assert any(f["name"] == "my_cases.csv" for f in files)

    read_req = urllib.request.Request(f"{server_url}/api/csv/read?file=my_cases.csv")
    with urllib.request.urlopen(read_req) as response:
        assert response.status == 200
        content = json.loads(response.read().decode("utf-8"))
        assert "headers" in content
        assert "rows" in content
        assert "input_text" in content["headers"]
        # Verify automatic baseline metric enrichment on raw test cases
        assert "baseline_accuracy" in content["headers"]
        assert "baseline_f1" in content["headers"]
        assert len(content["rows"]) >= 30
        first_row = content["rows"][0]
        assert float(first_row["baseline_accuracy"]) > 0.0
        assert float(first_row["baseline_f1"]) > 0.0


def test_api_model_train(server_url):
    payload = json.dumps({"text": "Astronauts explore galaxies and exoplanets."}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/model/train",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert data["tokens_learned"] > 0
        assert data["vocab_size"] > 0


def test_api_benchmark_run(server_url):
    payload = json.dumps({
        "cases_file": "my_cases.csv",
        "output_file": "test_api_benchmark_out.csv",
        "models": []
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/benchmark/run",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert data["success"] is True
        assert data["cases_processed"] > 0
        assert data["output_file"] == "test_api_benchmark_out.csv"
