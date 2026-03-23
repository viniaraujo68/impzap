"""
Shared card constants and utility functions for Truco Paulista.

All agents and the forward model import from here to avoid duplication.
Card strings use the canonical format "RANK_SUIT" (e.g. "3_CLUBS").
"""

from typing import Any, Dict, List

RANKS: List[str] = ["4", "5", "6", "7", "Q", "J", "K", "A", "2", "3"]
SUITS: List[str] = ["CLUBS", "SPADES", "HEARTS", "DIAMONDS"]
ALL_CARDS: List[str] = [f"{r}_{s}" for r in RANKS for s in SUITS]

# Suit strength order for manilha tie-breaking (higher = stronger).
MANILHA_SUIT_POWER: dict = {
    "DIAMONDS": 1,
    "SPADES": 2,
    "HEARTS": 3,
    "CLUBS": 4,
}

# Strength threshold above which a card is considered "strong".
# Rank index of "3" is 9; manilhas start at 10.
STRONG_CARD_THRESHOLD: int = 9


def card_strength(card: str, vira: str) -> int:
    """
    Return the numeric strength of a card given the current vira.

    Regular cards return their rank index [0..9] (Four=0, Three=9).
    Manilhas return 10 + suit_power [11..14] (Diamonds=11, Clubs=14).
    Face-down or unknown cards return -1.

    Parameters
    ----------
    card : str
        Card string ("RANK_SUIT"), "FACEDOWN", or the internal "FD:RANK_SUIT"
        marker used by the MCTS forward model.
    vira : str
        The vira card string.

    Returns
    -------
    int
        Numeric strength in [-1..14].
    """
    if not card or card == "FACEDOWN" or card.startswith("FD:"):
        return -1
    rank, suit = card.split("_")
    vira_rank = vira.split("_")[0]
    vira_idx = RANKS.index(vira_rank)
    manilha_rank = RANKS[(vira_idx + 1) % 10]
    if rank == manilha_rank:
        return 10 + MANILHA_SUIT_POWER[suit]
    return RANKS.index(rank)


def is_manilha(card: str, vira: str) -> bool:
    """
    Return True if card is the manilha for the given vira.

    The manilha is the rank immediately following the vira's rank on the
    circular RANKS sequence.
    """
    if not card or card == "FACEDOWN" or card.startswith("FD:"):
        return False
    rank = card.split("_")[0]
    vira_rank = vira.split("_")[0]
    vira_idx = RANKS.index(vira_rank)
    return rank == RANKS[(vira_idx + 1) % 10]


def card_to_go(card: str) -> Dict[str, Any]:
    """
    Convert a canonical card string ("RANK_SUIT") to the Go engine's JSON dict
    representation ``{"rank": int, "suit": int, "facedown": bool}``.

    Face-down card strings ("FACEDOWN" or "FD:RANK_SUIT") are encoded with
    ``facedown=True``; the underlying rank/suit are preserved when available
    (FD: prefix), otherwise rank=0/suit=0 are used as a placeholder.

    Parameters
    ----------
    card : str
        Card string in canonical format.

    Returns
    -------
    Dict[str, Any]
        ``{"rank": int, "suit": int, "facedown": bool}``
    """
    if card == "FACEDOWN":
        return {"rank": 0, "suit": 0, "facedown": True}
    facedown = card.startswith("FD:")
    base = card[3:] if facedown else card
    rank_str, suit_str = base.split("_")
    return {
        "rank": RANKS.index(rank_str),
        "suit": SUITS.index(suit_str),
        "facedown": facedown,
    }


def go_to_card(go_card: Dict[str, Any]) -> str:
    """
    Convert a Go engine card dict ``{"rank": int, "suit": int, "facedown": bool}``
    to the canonical card string ("RANK_SUIT") or "FACEDOWN".

    Parameters
    ----------
    go_card : Dict[str, Any]
        Card dict as returned by ``StepFromState``.

    Returns
    -------
    str
        Canonical card string or ``"FACEDOWN"``.
    """
    if go_card.get("facedown", False):
        return "FACEDOWN"
    return f"{RANKS[go_card['rank']]}_{SUITS[go_card['suit']]}"


def compare_cards(c1: str, c2: str, vira: str) -> int:
    """
    Compare two cards and return 1 if c1 wins, -1 if c2 wins, 0 for a tie.

    Mirrors the Compare() function in trucolib.go. Face-down cards ("FACEDOWN"
    or the "FD:RANK_SUIT" internal marker) always lose against face-up cards;
    two face-down cards tie.

    Parameters
    ----------
    c1, c2 : str
        Card strings.
    vira : str
        The vira card string.

    Returns
    -------
    int
        1 if c1 wins, -1 if c2 wins, 0 if tied.
    """
    c1_fd = c1 == "FACEDOWN" or c1.startswith("FD:")
    c2_fd = c2 == "FACEDOWN" or c2.startswith("FD:")

    if c1_fd and c2_fd:
        return 0
    if c1_fd:
        return -1
    if c2_fd:
        return 1

    s1 = card_strength(c1, vira)
    s2 = card_strength(c2, vira)
    if s1 > s2:
        return 1
    if s1 < s2:
        return -1
    return 0
