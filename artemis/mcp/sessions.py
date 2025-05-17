"""
ARTEMIS MCP Session Management

Manages debate sessions for the MCP server with
persistence, caching, and cleanup capabilities.
"""

import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from artemis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SessionMetadata:
    """Metadata about a debate session."""

    session_id: str
    topic: str
    model: str
    rounds_total: int
    rounds_completed: int
    status: str
    created_at: datetime
    updated_at: datetime
    pro_position: str = "supports the proposition"
    con_position: str = "opposes the proposition"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "model": self.model,
            "rounds_total": self.rounds_total,
            "rounds_completed": self.rounds_completed,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "pro_position": self.pro_position,
            "con_position": self.con_position,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            topic=data["topic"],
            model=data["model"],
            rounds_total=data["rounds_total"],
            rounds_completed=data["rounds_completed"],
            status=data["status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            pro_position=data.get("pro_position", "supports the proposition"),
            con_position=data.get("con_position", "opposes the proposition"),
            tags=data.get("tags", []),
        )


@dataclass
class SessionSnapshot:
    """A snapshot of session state for persistence."""

    metadata: SessionMetadata
    transcript: list[dict[str, Any]]
    verdict: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "transcript": self.transcript,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSnapshot":
        """Create from dictionary."""
        return cls(
            metadata=SessionMetadata.from_dict(data["metadata"]),
            transcript=data["transcript"],
            verdict=data.get("verdict"),
        )


class SessionStore:
    """
    Persistent storage for debate sessions.

    Provides file-based persistence with optional
    in-memory caching for performance.

    Example:
        >>> store = SessionStore("/path/to/sessions")
        >>> await store.save(snapshot)
        >>> restored = await store.load(session_id)
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        cache_enabled: bool = True,
        max_cache_size: int = 100,
    ) -> None:
        """
        Initialize the session store.

        Args:
            storage_path: Directory for session files.
            cache_enabled: Whether to cache sessions in memory.
            max_cache_size: Maximum cached sessions.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self._cache: dict[str, SessionSnapshot] = {}

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "SessionStore initialized",
            storage_path=str(self.storage_path) if self.storage_path else "memory-only",
            cache_enabled=cache_enabled,
        )

    async def save(self, snapshot: SessionSnapshot) -> None:
        """
        Save a session snapshot.

        Args:
            snapshot: Session snapshot to save.
        """
        session_id = snapshot.metadata.session_id

        # Update cache
        if self.cache_enabled:
            self._cache[session_id] = snapshot
            self._evict_cache_if_needed()

        # Persist to file
        if self.storage_path:
            file_path = self.storage_path / f"{session_id}.json"
            data = snapshot.to_dict()
            async with asyncio.Lock():
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)

        logger.debug("Session saved", session_id=session_id)

    async def load(self, session_id: str) -> SessionSnapshot | None:
        """
        Load a session snapshot.

        Args:
            session_id: Session to load.

        Returns:
            Session snapshot or None if not found.
        """
        # Check cache first
        if self.cache_enabled and session_id in self._cache:
            return self._cache[session_id]

        # Load from file
        if self.storage_path:
            file_path = self.storage_path / f"{session_id}.json"
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                snapshot = SessionSnapshot.from_dict(data)

                # Update cache
                if self.cache_enabled:
                    self._cache[session_id] = snapshot
                    self._evict_cache_if_needed()

                return snapshot

        return None

    async def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session to delete.

        Returns:
            True if deleted, False if not found.
        """
        deleted = False

        # Remove from cache
        if session_id in self._cache:
            del self._cache[session_id]
            deleted = True

        # Remove file
        if self.storage_path:
            file_path = self.storage_path / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
                deleted = True

        logger.debug("Session deleted", session_id=session_id, found=deleted)
        return deleted

    async def list_sessions(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionMetadata]:
        """
        List session metadata.

        Args:
            status: Filter by status.
            limit: Maximum sessions to return.
            offset: Offset for pagination.

        Returns:
            List of session metadata.
        """
        sessions = []

        # List from storage
        if self.storage_path:
            for file_path in sorted(
                self.storage_path.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    metadata = SessionMetadata.from_dict(data["metadata"])

                    if status and metadata.status != status:
                        continue

                    sessions.append(metadata)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        "Failed to load session",
                        file=str(file_path),
                        error=str(e),
                    )
        else:
            # List from cache only
            for snapshot in self._cache.values():
                if status and snapshot.metadata.status != status:
                    continue
                sessions.append(snapshot.metadata)

        # Apply pagination
        return sessions[offset:offset + limit]

    async def cleanup(
        self,
        max_age_hours: int = 24,
        status: str | None = None,
    ) -> int:
        """
        Clean up old sessions.

        Args:
            max_age_hours: Maximum session age in hours.
            status: Only clean sessions with this status.

        Returns:
            Number of sessions cleaned up.
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=max_age_hours)
        cleaned = 0

        sessions = await self.list_sessions(status=status, limit=10000)
        for metadata in sessions:
            if metadata.updated_at < cutoff and await self.delete(metadata.session_id):
                cleaned += 1

        logger.info(
            "Session cleanup completed",
            cleaned=cleaned,
            max_age_hours=max_age_hours,
        )
        return cleaned

    def _evict_cache_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) > self.max_cache_size:
            # Remove oldest entry
            oldest_id = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].metadata.updated_at,
            )
            del self._cache[oldest_id]


class SessionManager:
    """
    High-level session management for the MCP server.

    Handles session lifecycle, state transitions,
    and coordination with the session store.

    Example:
        >>> manager = SessionManager()
        >>> session_id = await manager.create_session(
        ...     topic="AI ethics",
        ...     model="gpt-4o",
        ... )
        >>> await manager.update_round(session_id, transcript_data)
        >>> await manager.complete_session(session_id, verdict)
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        auto_cleanup: bool = True,
        cleanup_interval_hours: int = 1,
        max_session_age_hours: int = 24,
    ) -> None:
        """
        Initialize the session manager.

        Args:
            store: Session store (creates default if not provided).
            auto_cleanup: Whether to run automatic cleanup.
            cleanup_interval_hours: Hours between cleanups.
            max_session_age_hours: Maximum session age.
        """
        self.store = store or SessionStore()
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval_hours = cleanup_interval_hours
        self.max_session_age_hours = max_session_age_hours
        self._cleanup_task: asyncio.Task | None = None

        logger.info("SessionManager initialized")

    async def start(self) -> None:
        """Start the session manager."""
        if self.auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")

    async def stop(self) -> None:
        """Stop the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None
            logger.info("Session cleanup task stopped")

    async def create_session(
        self,
        topic: str,
        model: str,
        rounds: int = 3,
        pro_position: str = "supports the proposition",
        con_position: str = "opposes the proposition",
        tags: list[str] | None = None,
    ) -> str:
        """
        Create a new session.

        Args:
            topic: Debate topic.
            model: LLM model.
            rounds: Number of rounds.
            pro_position: Pro-side position.
            con_position: Con-side position.
            tags: Optional tags.

        Returns:
            New session ID.
        """
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        metadata = SessionMetadata(
            session_id=session_id,
            topic=topic,
            model=model,
            rounds_total=rounds,
            rounds_completed=0,
            status="active",
            created_at=now,
            updated_at=now,
            pro_position=pro_position,
            con_position=con_position,
            tags=tags or [],
        )

        snapshot = SessionSnapshot(
            metadata=metadata,
            transcript=[],
            verdict=None,
        )

        await self.store.save(snapshot)

        logger.info(
            "Session created",
            session_id=session_id,
            topic=topic[:50],
        )

        return session_id

    async def get_session(self, session_id: str) -> SessionSnapshot | None:
        """Get a session by ID."""
        return await self.store.load(session_id)

    async def update_round(
        self,
        session_id: str,
        round_data: list[dict[str, Any]],
    ) -> None:
        """
        Update session with round data.

        Args:
            session_id: Session to update.
            round_data: New round transcript entries.
        """
        snapshot = await self.store.load(session_id)
        if not snapshot:
            raise ValueError(f"Session {session_id} not found")

        snapshot.transcript.extend(round_data)
        snapshot.metadata.rounds_completed += 1
        snapshot.metadata.updated_at = datetime.now()

        await self.store.save(snapshot)

        logger.debug(
            "Session round updated",
            session_id=session_id,
            rounds_completed=snapshot.metadata.rounds_completed,
        )

    async def complete_session(
        self,
        session_id: str,
        verdict: dict[str, Any],
    ) -> None:
        """
        Mark a session as completed with verdict.

        Args:
            session_id: Session to complete.
            verdict: Verdict data.
        """
        snapshot = await self.store.load(session_id)
        if not snapshot:
            raise ValueError(f"Session {session_id} not found")

        snapshot.verdict = verdict
        snapshot.metadata.status = "completed"
        snapshot.metadata.updated_at = datetime.now()

        await self.store.save(snapshot)

        logger.info(
            "Session completed",
            session_id=session_id,
            verdict=verdict.get("decision"),
        )

    async def archive_session(self, session_id: str) -> None:
        """
        Archive a completed session.

        Args:
            session_id: Session to archive.
        """
        snapshot = await self.store.load(session_id)
        if not snapshot:
            raise ValueError(f"Session {session_id} not found")

        snapshot.metadata.status = "archived"
        snapshot.metadata.updated_at = datetime.now()

        await self.store.save(snapshot)

        logger.info("Session archived", session_id=session_id)

    async def list_sessions(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SessionMetadata]:
        """List sessions with optional filtering."""
        return await self.store.list_sessions(status=status, limit=limit)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return await self.store.delete(session_id)

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
                await self.store.cleanup(max_age_hours=self.max_session_age_hours)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup error", error=str(e))


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())[:8]


async def create_session_manager(
    storage_path: str | Path | None = None,
    **kwargs: Any,
) -> SessionManager:
    """
    Factory function to create and start a session manager.

    Args:
        storage_path: Path for persistent storage.
        **kwargs: Additional configuration.

    Returns:
        Started session manager.
    """
    store = SessionStore(storage_path=storage_path)
    manager = SessionManager(store=store, **kwargs)
    await manager.start()
    return manager
