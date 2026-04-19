from fastapi import APIRouter, Depends, Query

from app.api.deps import get_asset_qa_service
from app.services.asset_qa_service import AssetQAService

router = APIRouter(prefix="/api/v1/assets")


@router.get("/{symbol}/price")
def get_asset_price(symbol: str, asset_qa_service: AssetQAService = Depends(get_asset_qa_service)) -> dict:
    return asset_qa_service.get_price_snapshot(symbol.upper())


@router.get("/{symbol}/history")
def get_asset_history(
    symbol: str,
    days: int = Query(default=7, ge=1, le=365),
    asset_qa_service: AssetQAService = Depends(get_asset_qa_service),
) -> dict:
    return asset_qa_service.get_price_history(symbol.upper(), days)

