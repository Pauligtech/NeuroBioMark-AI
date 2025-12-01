import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class JSONMemoryStore:
    """
    Lightweight JSON-backed memory for:
      • chat sessions (session_memory.json)
      • longitudinal MRI summaries (longitudinal_memory.json)

    This gives you a clear "memory" story for the Kaggle/ADK writeup.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # directory that contains this file (agent_system/memory)
            base_dir = os.path.dirname(__file__)

        self.base_dir = base_dir
        self.session_path = os.path.join(base_dir, "session_memory.json")
        self.longitudinal_path = os.path.join(base_dir, "longitudinal_memory.json")

        # Ensure files exist
        self._ensure_json_file(self.session_path, default={})
        self._ensure_json_file(self.longitudinal_path, default={})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_json_file(path: str, default: Any) -> None:
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(default, f, indent=2)

    @staticmethod
    def _load_json(path: str) -> Any:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _save_json(path: str, data: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Session / chat memory
    # ------------------------------------------------------------------
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Return list of turns for a given session_id.
        Each turn: { "role": "user"|"assistant", "content": str, "timestamp": iso }
        """
        all_sessions = self._load_json(self.session_path)
        return all_sessions.get(session_id, [])

    def append_session_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        all_sessions = self._load_json(self.session_path)
        history = all_sessions.get(session_id, [])

        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if extra:
            turn.update(extra)

        history.append(turn)
        all_sessions[session_id] = history
        self._save_json(self.session_path, all_sessions)

    # ------------------------------------------------------------------
    # Longitudinal MRI memory
    # ------------------------------------------------------------------
    def get_longitudinal(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """
        Return cached longitudinal summary for a subject, if any.
        """
        all_long = self._load_json(self.longitudinal_path)
        return all_long.get(subject_id)

    def put_longitudinal(self, subject_id: str, summary: Dict[str, Any]) -> None:
        """
        Store/overwrite longitudinal summary for a subject.
        Expected structure like:
            { "subject_id": "S001", "history": [ {...}, {...} ] }
        """
        all_long = self._load_json(self.longitudinal_path)
        all_long[subject_id] = summary
        self._save_json(self.longitudinal_path, all_long)


# ----------------------------------------------------------------------
# Convenience singleton accessor
# ----------------------------------------------------------------------

_GLOBAL_STORE: Optional[JSONMemoryStore] = None


def get_memory_store() -> JSONMemoryStore:
    """
    Return a global JSONMemoryStore instance so different modules
    can share the same memory without re-wiring everything.
    """
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = JSONMemoryStore()
    return _GLOBAL_STORE
