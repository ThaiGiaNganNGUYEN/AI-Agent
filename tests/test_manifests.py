import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_manifests_parse():
    paths = list((ROOT / "infra").rglob("*.yml")) + list((ROOT / "infra").rglob("*.yaml"))
    paths.append(ROOT / "docker-compose.yml")
    assert paths
    for path in paths:
        loaded = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        assert loaded, path


def test_grafana_dashboard_is_json():
    dashboard = json.loads((ROOT / "infra" / "grafana" / "dashboards" / "agentflow.json").read_text(encoding="utf-8"))
    assert dashboard["title"] == "AgentFlow"
    assert dashboard["panels"]
