from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
_DASH = Path(__file__).resolve().parent.parent.parent / "dashboard"


@router.get("/dashboard")
async def dashboard_index():
    return FileResponse(_DASH / "index.html", media_type="text/html")


@router.get("/dashboard/app.js")
async def dashboard_js():
    return FileResponse(_DASH / "app.js", media_type="application/javascript")


@router.get("/dashboard/style.css")
async def dashboard_css():
    return FileResponse(_DASH / "style.css", media_type="text/css")
