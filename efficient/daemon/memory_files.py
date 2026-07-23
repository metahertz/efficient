"""Server-side implementation of the Anthropic memory tool commands
(memory_20250818) backed by the memory_files collection, with content
embeddings for vector recall. Errors are returned as strings — they are
content for the model, not transport failures.
"""
import asyncio
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from efficient.db.collections import MEMORY_FILES
from efficient.modules.embeddings import embed_documents

_PREFIX = "/memories"
_EMBED_CHARS = 2000


class MemoryToolError(Exception):
    pass


def _validate(path: str) -> str:
    if not path.startswith(_PREFIX):
        raise MemoryToolError(f"path must start with {_PREFIX}: {path}")
    if ".." in path.split("/"):
        raise MemoryToolError(f"path traversal not allowed: {path}")
    return path.rstrip("/") or _PREFIX


async def _embed(content: str) -> list[float]:
    vecs = await asyncio.to_thread(embed_documents, [content[:_EMBED_CHARS]])
    return vecs[0]


async def _get(db, agent_id: str, path: str) -> dict | None:
    return await db[MEMORY_FILES].find_one({"agent_id": agent_id, "path": path})


async def _put(db, agent_id: str, path: str, content: str) -> None:
    now = datetime.now(timezone.utc)
    await db[MEMORY_FILES].update_one(
        {"agent_id": agent_id, "path": path},
        {"$set": {"content": content, "embedding": await _embed(content),
                  "updated_at": now},
         "$setOnInsert": {"created_at": now}},
        upsert=True,
    )


def _numbered(content: str, view_range: list | None) -> str:
    lines = content.splitlines()
    start, end = 1, len(lines)
    if view_range and len(view_range) == 2:
        start = max(1, int(view_range[0]))
        end = min(len(lines), int(view_range[1]))
    return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))


async def execute(db: AsyncIOMotorDatabase, agent_id: str, command: str,
                  args: dict) -> str:
    if command == "clear_all":
        result = await db[MEMORY_FILES].delete_many({"agent_id": agent_id})
        return f"All memory cleared ({result.deleted_count} files)"

    if command == "rename":
        old = _validate(args.get("old_path", ""))
        new = _validate(args.get("new_path", ""))
        doc = await _get(db, agent_id, old)
        if doc:
            await db[MEMORY_FILES].update_one(
                {"_id": doc["_id"]},
                {"$set": {"path": new, "updated_at": datetime.now(timezone.utc)}})
            return f"Renamed {old} to {new}"
        moved = 0
        async for d in db[MEMORY_FILES].find({"agent_id": agent_id,
                                              "path": {"$regex": f"^{old}/"}}):
            await db[MEMORY_FILES].update_one(
                {"_id": d["_id"]},
                {"$set": {"path": new + d["path"][len(old):]}})
            moved += 1
        if not moved:
            raise MemoryToolError(f"not found: {old}")
        return f"Renamed {old} to {new} ({moved} files)"

    path = _validate(args.get("path", ""))

    if command == "view":
        doc = await _get(db, agent_id, path)
        if doc:
            return _numbered(doc["content"], args.get("view_range"))
        prefix = "" if path == _PREFIX else path
        entries = [d["path"] async for d in db[MEMORY_FILES].find(
            {"agent_id": agent_id, "path": {"$regex": f"^{prefix or _PREFIX}/"}},
            {"path": 1}).sort("path", 1)]
        if not entries:
            raise MemoryToolError(f"not found: {path}")
        return "\n".join(entries)

    if command == "create":
        await _put(db, agent_id, path, args.get("file_text", ""))
        return f"File created successfully at {path}"

    if command == "str_replace":
        doc = await _get(db, agent_id, path)
        if not doc:
            raise MemoryToolError(f"not found: {path}")
        old_str = args.get("old_str", "")
        count = doc["content"].count(old_str)
        if count != 1:
            raise MemoryToolError(
                f"old_str must occur exactly once, found {count} occurrences")
        await _put(db, agent_id, path,
                   doc["content"].replace(old_str, args.get("new_str", "")))
        return f"File {path} edited"

    if command == "insert":
        doc = await _get(db, agent_id, path)
        if not doc:
            raise MemoryToolError(f"not found: {path}")
        lines = doc["content"].splitlines()
        at = int(args.get("insert_line", 0))
        if not 0 <= at <= len(lines):
            raise MemoryToolError(f"insert_line out of range: {at}")
        lines.insert(at, args.get("insert_text", "").rstrip("\n"))
        await _put(db, agent_id, path, "\n".join(lines))
        return f"Text inserted at line {at} in {path}"

    if command == "delete":
        result = await db[MEMORY_FILES].delete_many(
            {"agent_id": agent_id,
             "$or": [{"path": path}, {"path": {"$regex": f"^{path}/"}}]})
        if not result.deleted_count:
            raise MemoryToolError(f"not found: {path}")
        return f"Deleted {path}"

    raise MemoryToolError(f"unknown command: {command}")
