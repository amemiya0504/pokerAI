import pytest

from backend.engine.cards import Deck, card_str, derive_seed, parse_card, parse_cards


def test_card_roundtrip():
    for c in range(52):
        assert parse_card(card_str(c)) == c
    assert parse_cards("As Kd2c") == (parse_card("As"), parse_card("Kd"), parse_card("2c"))


@pytest.mark.parametrize("bad", ["A", "1s", "Ax", "AsK"])
def test_parse_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_cards(bad) if len(bad) != 2 else parse_card(bad)


def test_same_seed_same_deck():
    assert Deck(42).deal(52) == Deck(42).deal(52)
    assert Deck(42).deal(52) != Deck(43).deal(52)
    assert sorted(Deck(7).deal(52)) == list(range(52))


def test_deck_exhaustion():
    d = Deck(1)
    d.deal(52)
    with pytest.raises(ValueError):
        d.deal(1)


def test_derive_seed_is_stable_and_independent():
    assert derive_seed(1, "deck", 3) == derive_seed(1, "deck", 3)
    assert derive_seed(1, "deck", 3) != derive_seed(1, "deck", 4)
    assert derive_seed(1, "deck", 3) != derive_seed(1, "agent", 3)
