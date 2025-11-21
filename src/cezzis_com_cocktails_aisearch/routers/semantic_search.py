from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("v1/cocktails/search")
async def search():
    return {"message": "Search endpoint"}

