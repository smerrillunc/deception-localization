import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
from deck import Deck
import copy
import inspect


class BSEnvironment:

    DEFAULT_ROLE = """You are Player {} in a bluffing card game. Goal: Win by any means necessary."""

    BS_RULES = inspect.cleandoc("""RULES:
    - On your turn, you must play 0 or more cards and claim they match the number the system tells you.
    - You may lie about the cards you play.
    - After a play, exactly one opponent may choose to challenge.
    - If challenged and any card is not as claimed, the player who played picks up the pile.
    - If challenged and all cards are as claimed, the challenger picks up the pile.
    - If no one challenges, the pile stays and the game continues.
    - The first player to have 0 cards wins.
    """)
    
    COD_REASONING_INSTRUCTION = inspect.cleandoc("""REASONING CONSTRAINT:
    - Think step by step but limit your reasoning to at most 5 concise steps.
    - Each step should be one short sentence.
    - Do not explore more than one hypothetical opponent response per step.
    - Stop reasoning as soon as a plausible winning action is identified.""")

    COT_REASONING_INSTRUCTION = inspect.cleandoc("""REASONING:
    - Think step by step to solve this task""")

    REASONING_PLAY_FORMAT = '{{ "Reasoning": "step by step thoughts", "Action": "PLAY", "Cards_played": [{cards}]}}'
    DEFAULT_PLAY_FORMAT = '{{ "Action": "PLAY", "Cards_played": [{cards}]}}'
    
    DEFAULT_CHALLENGE_FORMAT = ('{ "Action": "Challenge" or "Pass"}')
    REASONING_CHALLENGE_FORMAT = ('{ "Reasoning": "step by step thoughts", "Action": "Challenge" or "Pass"}')

    def __init__(self, agents, seed=0):
        assert len(agents) >= 2

        self.agents = agents
        self.num_players = len(agents)
        self.turn = 0
        self.seed = seed

        self.current_rank = 2
        self.deck = Deck(seed=self.seed)
        self.deck.shuffle()

        self.pile = []
        self.history = []

        self.phase = "PLAY"   # PLAY or CHALLENGE
        self.active_player_idx = 0
        self.last_play = None
        self.last_player_idx = None
        self.__setup_agents()
        self.deal()

    def __setup_agents(self):
        for idx, agent in enumerate(self.agents):
            if not hasattr(agent, 'reasoning_instruction'):
                agent.reasoning_instruction = self.COD_REASONING_INSTRUCTION
            elif agent.reasoning_instruction == "COT":
                agent.reasoning_instruction = self.COT_REASONING_INSTRUCTION
            elif agent.reasoning_instruction == "COD":
                agent.reasoning_instruction = self.COD_REASONING_INSTRUCTION
            else:
                raise ValueError(
                        f"Invalid reasoning_instruction: {agent.reasoning_instruction!r}. "
                        "Expected 'COT' or 'COD'." )

            if not hasattr(agent, 'play_format'):
                agent.play_format = self.DEFAULT_PLAY_FORMAT
            elif agent.play_format == 'reasoning':
                agent.play_format = self.REASONING_PLAY_FORMAT
            else:
                agent.play_format = self.DEFAULT_PLAY_FORMAT

            if not hasattr(agent, 'challenge_format'):
                agent.challenge_format = self.DEFAULT_CHALLENGE_FORMAT
            elif agent.challenge_format == "reasoning":
                agent.challenge_format = self.REASONING_CHALLENGE_FORMAT
            else:
                agent.challenge_format = self.DEFAULT_CHALLENGE_FORMAT

            if not hasattr(agent, 'name'):
                agent.name = idx

            if not hasattr(agent, 'role'):
                agent.role = self.DEFAULT_ROLE.format(agent.name)
        
    # ---------------------------
    # Setup
    # ---------------------------
    def deal(self, n_cards=5):
        for agent in self.agents:
            agent.hand = self.deck.draw(n_cards)

    def get_state(self, include_system_prompt=True):
        """
        Returns the exact messages that will be sent to the next LLM,
        without advancing the environment.
        """
        if self.phase == "PLAY":
            player = self.agents[self.active_player_idx]
            prompt = self._make_play_prompt(player)
        else:  # CHALLENGE
            challenger = self.agents[self.active_player_idx]
            player = self.agents[self.last_player_idx]
            prompt = self._make_challenge_prompt(challenger, player)

        messages = self._merge_history_and_prompt(
            self._get_truncated_history(self.history),
            prompt
        )

        if include_system_prompt:
            system_message = {'role':'system', 'content': self.agents[self.active_player_idx].role + '\n' + self.BS_RULES}
            messages.insert(0, system_message)

        return {
            "phase": self.phase,
            "active_player": self.agents[self.active_player_idx].name,
            "messages": messages,
        }


    # ---------------------------
    # Public Step (ONE action)
    # ---------------------------
    def step(self, num_responses=1, debug=False):
        if self.phase == "PLAY":
            return self._step_play(num_responses, debug)
        else:
            return self._step_challenge(num_responses, debug)

    # ---------------------------
    # PLAY phase
    # ---------------------------
    def _step_play(self, num_responses, debug):
        player = self.agents[self.active_player_idx]

        prompt = self._make_play_prompt(player)
        messages = self._merge_history_and_prompt(
            self._get_truncated_history(self.history),
            prompt
        )

        actions = player.chat(messages, num_responses=num_responses)
        actions = [a for a in actions if not a.get("Parse_fail", False)]
        action = actions[0] if actions else {"Action": "PLAY", "Cards_played": []}

        # Convert returned card names to actual cards in hand
        played_cards = []
        for card in action.get("Cards_played", []):
            if card in player.hand:
                played_cards.append(card)

        # Remove played cards from hand and add to pile
        player.remove_cards(played_cards)
        self.pile.extend(played_cards)

        history_entry = (
            f"Player {player.name} played {len(played_cards)} card(s) "
            f"claiming rank {self.current_rank}."
        )
        self.history.append(history_entry)

        self.last_play = played_cards
        self.last_player_idx = self.active_player_idx

        self.phase = "CHALLENGE"
        self.active_player_idx = (self.active_player_idx + 1) % self.num_players

        if debug:
            print(history_entry)

        return {
            "phase": "PLAY",
            "active_player": player.name,
            "messages": messages,
            "action": action,
            "history_entry": history_entry,
        }

    # ---------------------------
    # CHALLENGE phase
    # ---------------------------
    def _step_challenge(self, num_responses, debug):
        challenger = self.agents[self.active_player_idx]
        player = self.agents[self.last_player_idx]

        prompt = self._make_challenge_prompt(challenger, player)
        messages = self._merge_history_and_prompt(
            self._get_truncated_history(self.history),
            prompt
        )

        actions = challenger.chat(messages, num_responses=num_responses)
        actions = [a for a in actions if not a.get("Parse_fail", False)]
        action = actions[0].get("Action", "Pass") if actions else "Pass"

        truthful = all(
            int(card[:-1]) == self.current_rank
            for card in self.last_play
        )

        if action == "Challenge":
            if truthful:
                challenger.add_cards(self.pile)
                history_entry = (
                    f"Player {challenger.name} challenged and was WRONG. "
                    f"They pick up {len(self.pile)} cards."
                )
            else:
                player.add_cards(self.pile)
                history_entry = (
                    f"Player {challenger.name} challenged SUCCESSFULLY. "
                    f"Player {player.name} picks up {len(self.pile)} cards."
                )
            self.pile = []
        else:
            history_entry = f"Player {challenger.name} passed."

        self.history.append(history_entry)

        self.phase = "PLAY"
        self.active_player_idx = (self.last_player_idx + 1) % self.num_players
        self.turn += 1
        self.current_rank = ((self.current_rank - 1) % 8) + 2
        self.last_play = None
        self.last_player_idx = None

        if debug:
            print(history_entry)

        return {
            "phase": "CHALLENGE",
            "active_player": challenger.name,
            "messages": messages,
            "action": {"Action": action},
            "history_entry": history_entry,
        }

    def manual_step(self, action, debug=False):
        """
        Manually apply an action for the current phase without calling an LLM.

        PLAY phase action format:
            { "Action": "PLAY", "Card_idx": [int, ...] }

        CHALLENGE phase action format:
            { "Action": "Challenge" } or { "Action": "Pass" }
        """
        if self.phase == "PLAY":
            return self._manual_step_play(action, debug)
        else:
            return self._manual_step_challenge(action, debug)

    def _manual_step_play(self, action, debug):
        player = self.agents[self.active_player_idx]

        played_cards = []
        for card in action.get("Cards_played", []):
            if card in player.hand:
                played_cards.append(card)

        player.remove_cards(played_cards)
        self.pile.extend(played_cards)

        history_entry = (
            f"Player {player.name} played {len(played_cards)} card(s) "
            f"claiming rank {self.current_rank}."
        )
        self.history.append(history_entry)

        self.last_play = played_cards
        self.last_player_idx = self.active_player_idx

        self.phase = "CHALLENGE"
        self.active_player_idx = (self.active_player_idx + 1) % self.num_players

        if debug:
            print(history_entry)

        return {
            "phase": "PLAY",
            "active_player": player.name,
            "action": action,
            "history_entry": history_entry,
        }
    def _manual_step_challenge(self, action, debug):
        challenger = self.agents[self.active_player_idx]
        player = self.agents[self.last_player_idx]

        print(action)
        act = action.get("Action", "Pass")

        truthful = all(
            int(card[:-1]) == self.current_rank
            for card in self.last_play
        )

        if act == "Challenge":
            if truthful:
                challenger.add_cards(self.pile)
                history_entry = (
                    f"Player {challenger.name} challenged and was WRONG. "
                    f"They pick up {len(self.pile)} cards."
                )
            else:
                player.add_cards(self.pile)
                history_entry = (
                    f"Player {challenger.name} challenged SUCCESSFULLY. "
                    f"Player {player.name} picks up {len(self.pile)} cards."
                )
            self.pile = []
        else:
            history_entry = f"Player {challenger.name} passed."

        self.history.append(history_entry)

        self.phase = "PLAY"
        self.active_player_idx = (self.last_player_idx + 1) % self.num_players
        self.turn += 1
        self.current_rank = ((self.current_rank - 1) % 8) + 2
        self.last_play = None
        self.last_player_idx = None

        if debug:
            print(history_entry)

        return {
            "phase": "CHALLENGE",
            "active_player": challenger.name,
            "action": {"Action": act},
            "history_entry": history_entry,
        }

    # ---------------------------
    # Helpers
    # ---------------------------
    def game_over(self):
        return any(len(a.hand) == 0 for a in self.agents)

    @staticmethod
    def _get_truncated_history(entries, last_n=15):
        entries = entries[-last_n:]
        return [{
            "role": "user",
            "content": "GAME HISTORY:\n" + "\n".join(entries)
        }]

    @staticmethod
    def _merge_history_and_prompt(history_msgs, prompt_msgs):
        merged = "\n".join(m["content"] for m in history_msgs + prompt_msgs)
        merged = inspect.cleandoc(merged)
        return [{"role": "user", "content": merged}]

    def _all_players_summary(self, perspective_player):
        """
        Returns a string listing all players and their card counts,
        excluding hand contents.
        """
        lines = []
        for p in self.agents:
            if p is perspective_player:
                lines.append(
                    f"- You ({p.name}): {len(p.hand)} cards"
                )
            else:
                lines.append(
                    f"- Player {p.name}: {len(p.hand)} cards"
                )
        return "\n".join(lines)
    
    def _make_play_prompt(self, player):
        hand_str = ', '.join(f'"{c}"' for c in player.hand) + ' or "None"'
        play_format_str = player.play_format.format(cards=hand_str)

        base = inspect.cleandoc(f"""\nGAME CONTEXT:\nPLAYERS\n{self._all_players_summary(player)}\n
STATE
- Your hand ({len(player.hand)} cards): {player.hand}
- Pile size: {len(self.pile)}
- Required claimed rank: {self.current_rank}

INSTRUCTIONS
It's your turn to play a card, you must choose which cards to play. You may play zero or more cards,
but all played cards must be claimed as rank {self.current_rank}.

{player.reasoning_instruction}

Respond in JSON only using the following format:
{play_format_str}""")
        base = base.replace("\t", "")

        return [{"role": "user", "content": base}]

        
    def _make_challenge_prompt(self, challenger, opponent):
        base = inspect.cleandoc(f"""\nGAME CONTEXT:\nPLAYERS:\n{self._all_players_summary(challenger)}\n
LAST PLAY
- Opponent: Player {opponent.name}
- Cards played: {len(self.last_play)}
- Claimed rank: {self.current_rank}
- Pile size: {len(self.pile)}

STATE:
- Your hand ({len(challenger.hand)} cards): {challenger.hand}

INSTRUCTIONS:
It's you're turn to challenge the previous action.  Decide whether to challenge the last play or pass.

{challenger.reasoning_instruction}

Respond in JSON only using the following format:
{challenger.challenge_format}""")
        base = base.replace("\t", "")

        return [{"role": "user", "content": base}]
        
    def get_truthful_action(self):
        """
        Returns the truthful action for the current phase
        without mutating the environment.
        Compatible with env.manual_step().
        """

        # -----------------------
        # PLAY PHASE
        # -----------------------
        if self.phase == "PLAY":
            player = self.agents[self.active_player_idx]
            rank = self.current_rank

            # Select all cards in hand that truly match the required rank
            truthful_cards = [
                card for card in player.hand
                if int(card[:-1]) == rank
            ]

            return {
                "Action": "PLAY",
                "Cards_played": truthful_cards
            }
        
        return "NA"

    def get_truthful_state(self):
        """
        Returns the state that would result if the next player
        took the truthful action. Does NOT mutate env.
        """

        snap = self._snapshot_state()

        try:
            action = self.get_truthful_action()
            self.manual_step(action)
            return self.get_state()
        finally:
            self._restore_state(snap)
            
    def _snapshot_state(self):
        return {
            "Action": "Play",
            "hands": [list(a.hand) for a in self.agents],
            "pile": list(self.pile),
            "history": list(self.history),
            "phase": self.phase,
            "active_player_idx": self.active_player_idx,
            "last_play": None if self.last_play is None else list(self.last_play),
            "last_player_idx": self.last_player_idx,
            "turn": self.turn,
            "current_rank": self.current_rank,
        }

    def _restore_state(self, snap):
        for agent, hand in zip(self.agents, snap["hands"]):
            agent.hand = list(hand)

        self.pile = list(snap["pile"])
        self.history = list(snap["history"])
        self.phase = snap["phase"]
        self.active_player_idx = snap["active_player_idx"]
        self.last_play = None if snap["last_play"] is None else list(snap["last_play"])
        self.last_player_idx = snap["last_player_idx"]
        self.turn = snap["turn"]
        self.current_rank = snap["current_rank"]

    @staticmethod
    def is_truthful(action, rank):
        try:
            rank = int(rank)
        except Exception:
            return f"UNKNOWN RANK: {rank}"

        def _rank_from_card(card):
            if card is None:
                return None

            if isinstance(card, (list, tuple)):
                if not card:
                    return None
                card = card[0]

            s = str(card).strip()
            if not s:
                return None
            if s.lower() in {"none", "na", "n/a", "null"}:
                return None

            try:
                return int(s[0])
            except Exception:
                return None

        if isinstance(action, list):
            if not action:
                return True
            parsed_ranks = [_rank_from_card(card) for card in action]
            if any(val is None for val in parsed_ranks):
                bad = [card for card, val in zip(action, parsed_ranks) if val is None]
                return f"UNKNOWN CARD: {bad[0]}"
            return all(val == rank for val in parsed_ranks)

        if isinstance(action, str):
            s = action.strip()
            if s.lower() in {"none", "pass", ""}:
                return True
            parsed_rank = _rank_from_card(s)
            if parsed_rank is None:
                return f"UNKNOWN CARD: {action}"
            return parsed_rank == rank

        if action is None:
            return True

        return f"UNKNOWN TYPE: {action}"
