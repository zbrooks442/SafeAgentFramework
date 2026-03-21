from uuid import UUID

from safe_agent.core import SessionManager


def test_create_session() -> None:
    manager = SessionManager()

    session_one = manager.create()
    session_two = manager.create()

    assert session_one.id != session_two.id
    assert UUID(session_one.id)
    assert UUID(session_two.id)


def test_get_session() -> None:
    manager = SessionManager()
    session = manager.create()

    assert manager.get(session.id) is session


def test_close_session() -> None:
    manager = SessionManager()
    session = manager.create()

    manager.close(session.id)

    assert manager.get(session.id) is None


def test_list_active() -> None:
    manager = SessionManager()
    session_one = manager.create()
    session_two = manager.create()

    assert manager.list_active() == [session_one.id, session_two.id]


def test_session_isolation() -> None:
    manager = SessionManager()
    session_one = manager.create()
    session_two = manager.create()

    session_one.messages.append({"role": "user", "content": "one"})
    session_two.messages.append({"role": "user", "content": "two"})

    assert session_one.messages == [{"role": "user", "content": "one"}]
    assert session_two.messages == [{"role": "user", "content": "two"}]
