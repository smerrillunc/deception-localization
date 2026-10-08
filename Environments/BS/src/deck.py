from itertools import product
import random

class Deck:
    def __init__(self, ranks=None, suits=None, seed=None):
        self.ranks = ranks or ['2', '3', '4', '5', '6', '7', '8', '9']
        self.suits = suits or ['h', 'd', 's', 'c']
        self.seed = seed
        self.rng = random.Random(seed)
        self.cards = self._build_deck()

    def _build_deck(self):
        return [f"{rank}{suit}" for rank, suit in product(self.ranks, self.suits)]

    def shuffle(self):
        self.rng.shuffle(self.cards)

    def draw(self, n=1):
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def reset(self, seed=None):
        """Rebuild and reshuffle the deck (optionally with a new seed)."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)
        self.cards = self._build_deck()
        self.shuffle()

    def __len__(self):
        return len(self.cards)

    def __repr__(self):
        return f"Deck({len(self.cards)} cards, seed={self.seed})"
