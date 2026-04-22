from typing import List, Dict, Optional, Any

import requests
from langchain_core.tools import tool

from app.settings import get_settings


def _shopify_headers() -> Dict[str, str]:
    return {"X-Shopify-Access-Token": get_settings().shopify_access_token}


def _base_url() -> str:
    return get_settings().shopify_base_url.rstrip("/")


@tool
def get_all_products(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch all products from the Shopify store."""
    print(f"TOOL: get_all_products")
    url = f"{_base_url()}/products.json?limit={limit}"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        return response.json().get("products", [])
    except Exception as e:
        print(f"Error fetching products: {e}")
        return []


@tool
def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    """Get a single product by Shopify product ID."""
    print(f"TOOL: get_product_by_id")
    url = f"{_base_url()}/products/{product_id}.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        return response.json().get("product")
    except Exception as e:
        print(f"Error fetching product by ID: {e}")
        return None


@tool
def search_products_by_title(title: str) -> List[Dict[str, Any]]:
    """Search products by their title."""
    print(f"TOOL: search_products_by_title")
    url = f"{_base_url()}/products.json?title={title}"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        return response.json().get("products", [])
    except Exception as e:
        print(f"Error searching products by title: {e}")
        return []


@tool
def get_product_inventory(product_id: int) -> Optional[int]:
    """Get the total available inventory quantity for a product (sum of all variants)."""
    print(f"TOOL: get_product_inventory")
    url = f"{_base_url()}/products/{product_id}.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        product = response.json().get("product")
        if not product:
            return None
        return sum(
            variant.get("inventory_quantity", 0)
            for variant in product.get("variants", [])
        )
    except Exception as e:
        print(f"Error fetching product inventory: {e}")
        return None


@tool
def get_product_price(product_id: int) -> Optional[str]:
    """Get the price of a product (returns the first variant's price)."""
    print(f"TOOL: get_product_price")
    url = f"{_base_url()}/products/{product_id}.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        product = response.json().get("product")
        if not product:
            return None
        return product.get("variants", [{}])[0].get("price")
    except Exception as e:
        print(f"Error fetching product price: {e}")
        return None


@tool
def list_product_variants(product_id: int) -> List[Dict[str, Any]]:
    """List all variants of a given product."""
    print(f"TOOL: list_product_variants")
    url = f"{_base_url()}/products/{product_id}.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        product = response.json().get("product")
        if not product:
            return []
        return product.get("variants", [])
    except Exception as e:
        print(f"Error listing product variants: {e}")
        return []


@tool
def get_product_image_urls(product_id: int) -> List[str]:
    """Get all image URLs for a given product."""
    print(f"TOOL: get_product_image_urls")
    url = f"{_base_url()}/products/{product_id}.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        product = response.json().get("product")
        if not product:
            return []
        return [img.get("src") for img in product.get("images", []) if img.get("src")]
    except Exception as e:
        print(f"Error fetching product images: {e}")
        return []


PRODUCT_TOOL_FUNCTIONS = [
    get_all_products,
    get_product_by_id,
    search_products_by_title,
    get_product_inventory,
    get_product_price,
    list_product_variants,
    get_product_image_urls,
]
