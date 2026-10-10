from video_engine import perf


def test_learning_updates_rates(tmp_path, monkeypatch):
    monkeypatch.setenv("VE_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("VE_LEARN_PERF", "true")
    from video_engine import config
    config.get_settings.cache_clear()
    assert perf.rates()["sr_coreml_s_per_mp"] == perf.DEFAULTS["sr_coreml_s_per_mp"]
    plan = {"source_width": 1000, "source_height": 1000,
            "sr": {"enabled": True, "out_width": 4000, "out_height": 4000},
            "interpolation": {"enabled": False}}
    metrics = {"stage_seconds": {"super_resolution": 200.0}, "frames_decoded": 100, "frames_written": 100,
               "models": {"super_resolution": {"device": "coreml:neural-engine"}}}
    perf.learn(plan, metrics)  # observed 2.0 s/MP
    r = perf.rates()["sr_coreml_s_per_mp"]
    assert abs(r - (0.5 * 2.0 + 0.5 * perf.DEFAULTS["sr_coreml_s_per_mp"])) < 1e-3
    assert "learned from 1 job" in perf.basis()
    config.get_settings.cache_clear()
