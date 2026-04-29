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


def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    assert score_transaction(_base(amount_usd=1200)) >= 25


def test_high_device_risk_adds_risk():
    high = score_transaction(_base(device_risk_score=70))
    low = score_transaction(_base(device_risk_score=10))
    assert high > low


def test_international_adds_risk():
    intl = score_transaction(_base(is_international=1))
    domestic = score_transaction(_base(is_international=0))
    assert intl > domestic


def test_high_velocity_adds_risk():
    high_vel = score_transaction(_base(velocity_24h=6))
    low_vel = score_transaction(_base(velocity_24h=1))
    assert high_vel > low_vel


def test_prior_chargebacks_add_risk():
    two_cb = score_transaction(_base(prior_chargebacks=2))
    one_cb = score_transaction(_base(prior_chargebacks=1))
    no_cb = score_transaction(_base(prior_chargebacks=0))
    assert two_cb > one_cb > no_cb


def test_worst_case_is_high_risk():
    tx = _base(
        device_risk_score=80,
        is_international=1,
        amount_usd=1200,
        velocity_24h=8,
        failed_logins_24h=5,
        prior_chargebacks=2,
    )
    assert label_risk(score_transaction(tx)) == "high"
