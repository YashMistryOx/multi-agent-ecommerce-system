import re
from typing import List, Dict, Optional, Any

import requests
from langchain_core.tools import tool

from app.settings import get_settings


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


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


@tool
def recommend_products(query: str) -> List[Dict[str, Any]]:
    """Recommend products from the store that best match a customer's natural language request.

    Call this tool whenever the user:
    - Asks for a recommendation or suggestion of any kind
    - Describes a need, goal, problem, occasion, or use case
    - Mentions a budget, recipient, preference, or lifestyle
    - Uses words like 'suggest', 'recommend', 'looking for', 'need something', 'what do you have for'
    - Asks what's popular, best-selling, new, or trending

    Pass the user's message (or a concise rephrasing of their intent) as the query.
    Never recommend products from memory — always call this tool so results are based on
    the actual live store catalog.

    Returns a list of matched products, each with a 'recommendation_reason' field.
    """
    print(f"TOOL: recommend_products query={query!r}")

    # --- 1. Load all products with their details ---
    url = f"{_base_url()}/products.json?limit=100"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        products = response.json().get("products", [])
    except Exception as e:
        print(f"Error fetching products for recommendation: {e}")
        return []

    if not products:
        return []

    # --- 2. Build a compact catalog for the LLM to reason over ---
    catalog_lines = []
    for p in products:
        price = p.get("variants", [{}])[0].get("price", "N/A")
        description = _strip_html(p.get("body_html", ""))[:300]
        tags = p.get("tags", "")
        catalog_lines.append(
            f"ID:{p['id']} | {p['title']} | type:{p.get('product_type', '')} "
            f"| tags:{tags} | price:{price} | desc:{description}"
        )
    catalog_text = "\n".join(catalog_lines)

    # --- 3. Ask the LLM to pick the best matches and explain why ---
    from app.agents import llm

    prompt = (
        f"You are a helpful shopping assistant. A customer said: \"{query}\"\n\n"
        f"Below is the store's product catalog (one product per line):\n"
        f"{catalog_text}\n\n"
        f"Select the top 3–5 most relevant products for the customer's request. "
        f"Reply ONLY as a JSON array where each element has:\n"
        f'  "id": <product id as integer>,\n'
        f'  "reason": <one sentence explaining why this product fits the request>\n'
        f"Output only valid JSON, no markdown fences, no extra text."
    )

    try:
        raw = llm.invoke(prompt).content.strip()
        matches = __import__("json").loads(raw)
    except Exception as e:
        print(f"LLM recommendation parsing failed: {e}")
        return []

    # --- 4. Attach the reason to the full product object ---
    product_by_id = {p["id"]: p for p in products}
    results = []
    for match in matches:
        pid = match.get("id")
        if pid and pid in product_by_id:
            product = dict(product_by_id[pid])
            product["recommendation_reason"] = match.get("reason", "")
            results.append(product)

    return results


PRODUCT_TOOL_FUNCTIONS = [
    get_all_products,
    get_product_by_id,
    search_products_by_title,
    get_product_inventory,
    get_product_price,
    list_product_variants,
    get_product_image_urls,
    recommend_products,
]
