"""
Unit & Integration Tests for Persistent Flight Annotations & Anomaly Notes Engine.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from flightml.cli import app
from flightml.eval.flight_notes import (
    FlightNote,
    FlightNotesManager,
    NOTE_CATEGORIES,
    get_notes_manager,
)
from flightml.mcp.server import create_mcp_server

runner = CliRunner()


def test_seed_note_lh574_cape_town():
    """Verify that Lufthansa LH574 (292222833) Munich -> Cape Town overflight permission note is seeded and accurate."""
    mgr = get_notes_manager()
    note = mgr.get_note(292222833)

    assert note is not None
    assert isinstance(note, FlightNote)
    assert note.ectrl_id == 292222833
    assert "Lufthansa" in note.title
    assert "LH574" in note.title
    assert note.category == "Diplomatic / Overflight Rights"
    assert "aeroTELEGRAPH" in note.note
    assert "traffic rights" in note.note
    assert "winter" in note.note.lower()
    assert note.source_url is not None
    assert "aerotelegraph.com" in note.source_url

    # Check cached metadata
    m = note.cached_metadata
    assert m.get("adep") == "EDDM"
    assert m.get("ades") == "FACT"
    assert m.get("ac_operator") == "DLH"
    assert m.get("ac_type") == "A359"


def test_flight_notes_crud_isolated(tmp_path: Path):
    """Test full CRUD operations on an isolated temporary notes manager."""
    test_file = tmp_path / "test_notes.json"
    mgr = FlightNotesManager(notes_path=test_file)

    # Initial seed exists
    assert mgr.get_note(292222833) is not None

    # Create new note
    test_fid = 999123456
    saved = mgr.save_note(
        ectrl_id=test_fid,
        title="Test Severe Convective Storm Detour",
        note="Aircraft diverted 250 NM east to circumvent intense squall line over the Alps.",
        category="Weather Diversion / Severe Convection",
        author="Lead Meteorologist",
        tags=["convection", "squall", "alps", "thunderstorm"],
        source_url="https://aviation-weather.example.com",
    )
    assert saved.ectrl_id == test_fid
    assert saved.category == "Weather Diversion / Severe Convection"
    assert "squall" in saved.tags

    # Retrieve note
    retrieved = mgr.get_note(test_fid)
    assert retrieved is not None
    assert retrieved.title == "Test Severe Convective Storm Detour"
    assert retrieved.author == "Lead Meteorologist"

    # Search notes
    search_res = mgr.list_notes(search_query="squall")
    assert any(n.ectrl_id == test_fid for n in search_res)

    # Category filter
    cat_res = mgr.list_notes(category="Weather Diversion / Severe Convection")
    assert any(n.ectrl_id == test_fid for n in cat_res)
    assert not any(n.ectrl_id == 292222833 for n in cat_res)

    # Update note
    updated = mgr.save_note(
        ectrl_id=test_fid,
        title="Updated Storm Detour with Radar Confirmation",
        note="Radar confirms Tops FL420 over northern Italy.",
        category="Weather Diversion / Severe Convection",
        author="Chief Dispatcher",
        tags=["convection", "radar-confirmed"],
    )
    assert updated.title == "Updated Storm Detour with Radar Confirmation"
    assert updated.author == "Chief Dispatcher"
    assert "radar-confirmed" in updated.tags

    # Verify update persisted
    verify_note = mgr.get_note(test_fid)
    assert verify_note is not None
    assert verify_note.title == "Updated Storm Detour with Radar Confirmation"

    # Delete note
    deleted = mgr.delete_note(test_fid)
    assert deleted is True
    assert mgr.get_note(test_fid) is None


def test_cli_flight_notes_commands():
    """Verify CLI commands flight-notes-list, flight-notes-get, and flight-notes-set."""
    # 1. List notes as JSON
    res_list = runner.invoke(app, ["flight-notes-list", "--json"])
    assert res_list.exit_code == 0
    data = json.loads(res_list.stdout)
    assert isinstance(data, list)
    assert any(n["ectrl_id"] == 292222833 for n in data)

    # 2. Get specific seed note
    res_get = runner.invoke(app, ["flight-notes-get", "292222833", "--json"])
    assert res_get.exit_code == 0
    note = json.loads(res_get.stdout)
    assert note["ectrl_id"] == 292222833
    assert "Lufthansa" in note["title"]
    assert note["category"] == "Diplomatic / Overflight Rights"

    # 3. Create note via CLI
    temp_fid = 888777666
    res_set = runner.invoke(app, [
        "flight-notes-set",
        str(temp_fid),
        "--title", "CLI Test Volcanic Ash Detour",
        "--note", "Etna eruption NOTAM triggered rerouting via Tyrrhenian Sea.",
        "--category", "Airspace Closure / Conflict Zone Avoidance",
        "--author", "CLI Tester",
        "--tags", "etna, volcano, notam",
    ])
    assert res_set.exit_code == 0
    assert "successfully saved" in res_set.stdout

    # 4. Verify retrieval
    res_check = runner.invoke(app, ["flight-notes-get", str(temp_fid), "--json"])
    assert res_check.exit_code == 0
    c_note = json.loads(res_check.stdout)
    assert c_note["title"] == "CLI Test Volcanic Ash Detour"

    # 5. Delete via CLI
    res_del = runner.invoke(app, ["flight-notes-delete", str(temp_fid)])
    assert res_del.exit_code == 0
    assert "deleted successfully" in res_del.stdout


@pytest.mark.asyncio
async def test_mcp_flight_notes_tools():
    """Verify MCP tools get_flight_note, save_flight_note, and list_flight_notes."""
    server = create_mcp_server()

    # 1. get_flight_note for LH574
    res_raw = await server.call_tool("get_flight_note", {"flight_id": 292222833})
    res = json.loads(res_raw[0][0].text)
    assert res.get("found") is True
    assert res.get("ectrl_id") == 292222833
    assert "Lufthansa" in res.get("title", "")

    # 2. list_flight_notes
    list_raw = await server.call_tool("list_flight_notes", {"search": "Lufthansa"})
    l_res = json.loads(list_raw[0][0].text)
    assert l_res.get("total_notes", 0) >= 1

    # 3. save_flight_note
    mcp_fid = 777666555
    save_raw = await server.call_tool("save_flight_note", {
        "flight_id": mcp_fid,
        "title": "MCP Test Ground Stop Hold",
        "note": "Ground delay program at EHAM caused 45-minute holding over North Sea.",
        "category": "Slot Regulation / Curfew Management",
        "author": "MCP Agent",
        "tags": "eham, holding, slot",
    })
    s_res = json.loads(save_raw[0][0].text)
    assert s_res.get("saved") is True
    assert s_res["note"]["ectrl_id"] == mcp_fid

    # Clean up
    get_notes_manager().delete_note(mcp_fid)
