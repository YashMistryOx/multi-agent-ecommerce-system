from langchain.tools import tool
from typing import List, Dict, Any
import requests

from app.settings import get_settings

def _shopify_headers() -> Dict[str, str]:
    settings = get_settings()
    return {"X-Shopify-Access-Token": settings.shopify_access_token}

def _base_url() -> str:
    return get_settings().shopify_base_url.rstrip("/")

@tool
def get_all_orders(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch all orders from the Shopify store."""
    url = f"{_base_url()}/orders.json?limit={limit}"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        return response.json().get("orders", [])
    except Exception as e:
        print(f"Error fetching orders: {e}")
        return []

ORDERS_TOOL_FUNCTIONS = [
    get_all_orders,
]