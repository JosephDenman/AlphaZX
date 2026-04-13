"""Backward-compatible re-export — canonical location is alphazx.shared.game_state."""
from alphazx.shared.game_state import *  # noqa: F401,F403
from alphazx.shared.game_state import GameState, _clone_match_diagram  # noqa: F401 — explicit re-exports
