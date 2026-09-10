import pytest
from src.intent_classifier import MajorityClassClassifier, SimpleTfidfLogisticClassifier, CalibratedHybridClassifier

def test_majority_classifier():
    X = ["iOS update crash", "battery drain", "battery drain fast"]
    y = ["OS_UPDATE_BUG", "BATTERY_POWER", "BATTERY_POWER"]
    clf = MajorityClassClassifier().fit(X, y)
    preds = clf.predict(["any query"])
    assert preds[0] == "BATTERY_POWER"

def test_tfidf_logistic_classifier():
    X = ["iOS update update update", "battery drain battery power"]
    y = ["OS_UPDATE_BUG", "BATTERY_POWER"]
    clf = SimpleTfidfLogisticClassifier().fit(X, y)
    preds = clf.predict(["iOS update crash"])
    assert preds[0] in ["OS_UPDATE_BUG", "BATTERY_POWER"]

def test_calibrated_hybrid_classifier():
    X = ["iOS update bug crash", "battery drain fast power", "wifi connectivity dropping"]
    y = ["OS_UPDATE_BUG", "BATTERY_POWER", "CONNECTIVITY_WIFI_BT"]
    clf = CalibratedHybridClassifier().fit(X, y)
    res = clf.predict_single("My battery is dying fast")
    assert "intent" in res
    assert "intent_confidence" in res
    assert 0.0 <= res["intent_confidence"] <= 1.0
