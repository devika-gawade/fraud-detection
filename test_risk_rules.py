import pandas as pd
import pytest

from features import build_model_frame
from risk_rules import label_risk, score_transaction


def _base(**overrides):
    tx = {
        "device_risk_score": 0,
        "is_international": 0,
        "amount_usd": 0,
        "velocity_24h": 0,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk — exact boundary values
# ---------------------------------------------------------------------------

def test_label_risk_exact_boundaries():
    assert label_risk(0) == "low"
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"
    assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — exact point values per signal
# ---------------------------------------------------------------------------

def test_clean_transaction_scores_zero():
    assert score_transaction(_base()) == 0


def test_device_risk_high_tier():
    assert score_transaction(_base(device_risk_score=70)) == 25
    assert score_transaction(_base(device_risk_score=99)) == 25


def test_device_risk_medium_tier():
    assert score_transaction(_base(device_risk_score=40)) == 10
    assert score_transaction(_base(device_risk_score=69)) == 10


def test_device_risk_low_tier_adds_nothing():
    assert score_transaction(_base(device_risk_score=39)) == 0


def test_international_adds_15():
    assert score_transaction(_base(is_international=1)) == 15


def test_domestic_adds_nothing():
    assert score_transaction(_base(is_international=0)) == 0


def test_amount_high_tier():
    assert score_transaction(_base(amount_usd=1000)) == 25
    assert score_transaction(_base(amount_usd=5000)) == 25


def test_amount_medium_tier():
    assert score_transaction(_base(amount_usd=500)) == 10
    assert score_transaction(_base(amount_usd=999)) == 10


def test_amount_low_adds_nothing():
    assert score_transaction(_base(amount_usd=499)) == 0


def test_velocity_high_tier():
    assert score_transaction(_base(velocity_24h=6)) == 20
    assert score_transaction(_base(velocity_24h=10)) == 20


def test_velocity_medium_tier():
    assert score_transaction(_base(velocity_24h=3)) == 5
    assert score_transaction(_base(velocity_24h=5)) == 5


def test_velocity_low_adds_nothing():
    assert score_transaction(_base(velocity_24h=2)) == 0


def test_failed_logins_high_tier():
    assert score_transaction(_base(failed_logins_24h=5)) == 20
    assert score_transaction(_base(failed_logins_24h=10)) == 20


def test_failed_logins_medium_tier():
    assert score_transaction(_base(failed_logins_24h=2)) == 10
    assert score_transaction(_base(failed_logins_24h=4)) == 10


def test_failed_logins_low_adds_nothing():
    assert score_transaction(_base(failed_logins_24h=1)) == 0


def test_prior_chargebacks_two_or_more():
    assert score_transaction(_base(prior_chargebacks=2)) == 20
    assert score_transaction(_base(prior_chargebacks=5)) == 20


def test_prior_chargebacks_one():
    assert score_transaction(_base(prior_chargebacks=1)) == 5


def test_prior_chargebacks_none_adds_nothing():
    assert score_transaction(_base(prior_chargebacks=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — score clamping
# ---------------------------------------------------------------------------

def test_score_capped_at_100():
    # All signals at max: 25+15+25+20+20+20 = 125, must clamp to 100
    tx = _base(
        device_risk_score=80,
        is_international=1,
        amount_usd=1200,
        velocity_24h=8,
        failed_logins_24h=5,
        prior_chargebacks=2,
    )
    assert score_transaction(tx) == 100


def test_score_never_negative():
    assert score_transaction(_base()) >= 0


# ---------------------------------------------------------------------------
# score_transaction — combined signals produce expected risk labels
# ---------------------------------------------------------------------------

def test_low_risk_profile():
    tx = _base(amount_usd=50, device_risk_score=5)
    assert label_risk(score_transaction(tx)) == "low"


def test_medium_risk_profile():
    # $600 purchase on a medium-risk device, domestic, no history → 10+10 = 20...
    # need a bit more: add 1 prior chargeback → 10+10+5 = 25, still low
    # add medium velocity → 25+5 = 30 = medium
    tx = _base(amount_usd=600, device_risk_score=50, prior_chargebacks=1, velocity_24h=3)
    assert label_risk(score_transaction(tx)) == "medium"


def test_high_risk_profile():
    tx = _base(
        device_risk_score=80,
        is_international=1,
        amount_usd=1200,
        velocity_24h=8,
        failed_logins_24h=5,
        prior_chargebacks=2,
    )
    assert label_risk(score_transaction(tx)) == "high"


# ---------------------------------------------------------------------------
# build_model_frame — features.py
# ---------------------------------------------------------------------------

def _make_frames(amount_usd=100, failed_logins_24h=0):
    transactions = pd.DataFrame([{
        "transaction_id": 1,
        "account_id": 42,
        "amount_usd": amount_usd,
        "failed_logins_24h": failed_logins_24h,
    }])
    accounts = pd.DataFrame([{
        "account_id": 42,
        "customer_name": "Test User",
    }])
    return transactions, accounts


def test_build_model_frame_merges_account_columns():
    txns, accts = _make_frames()
    result = build_model_frame(txns, accts)
    assert "customer_name" in result.columns
    assert result.iloc[0]["customer_name"] == "Test User"


def test_is_large_amount_true_at_threshold():
    txns, accts = _make_frames(amount_usd=1000)
    result = build_model_frame(txns, accts)
    assert result.iloc[0]["is_large_amount"] == 1


def test_is_large_amount_false_below_threshold():
    txns, accts = _make_frames(amount_usd=999)
    result = build_model_frame(txns, accts)
    assert result.iloc[0]["is_large_amount"] == 0


def test_login_pressure_none():
    txns, accts = _make_frames(failed_logins_24h=0)
    result = build_model_frame(txns, accts)
    assert result.iloc[0]["login_pressure"] == "none"


def test_login_pressure_low():
    txns, accts = _make_frames(failed_logins_24h=1)
    result = build_model_frame(txns, accts)
    assert result.iloc[0]["login_pressure"] == "low"


def test_login_pressure_high():
    txns, accts = _make_frames(failed_logins_24h=5)
    result = build_model_frame(txns, accts)
    assert result.iloc[0]["login_pressure"] == "high"
