from typing import List, Dict, Optional, Any

import requests
from langchain_core.tools import tool

from app.settings import get_settings


def _shopify_headers() -> Dict[str, str]:
    return {"X-Shopify-Access-Token": get_settings().shopify_access_token}


def _base_url() -> str:
    return get_settings().shopify_base_url.rstrip("/")


@tool
def get_orders_by_email(email: str) -> List[Dict[str, Any]]:
    """Fetch all orders placed by a customer email address."""
    print(f"TOOL: get_orders_by_email")
    url = f"{_base_url()}/orders.json?email={email}&status=any"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        return response.json().get("orders", [])
    except Exception as e:
        print(f"Error fetching orders by email: {e}")
        return []


@tool
def get_order_by_id(order_id: int) -> Optional[Dict[str, Any]]:
    """Get full details of a specific order by its Shopify order ID."""
    print(f"TOOL: get_order_by_id")
    url = f"{_base_url()}/orders/{order_id}.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        return response.json().get("order")
    except Exception as e:
        print(f"Error fetching order by ID: {e}")
        return None


@tool
def get_order_status(order_id: int) -> Optional[Dict[str, str]]:
    """Get the financial and fulfillment status of a specific order."""
    print(f"TOOL: get_order_status")
    url = f"{_base_url()}/orders/{order_id}.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        order = response.json().get("order")
        if not order:
            return None
        return {
            "order_id": str(order.get("id")),
            "order_number": str(order.get("order_number")),
            "financial_status": order.get("financial_status", "unknown"),
            "fulfillment_status": order.get("fulfillment_status") or "unfulfilled",
        }
    except Exception as e:
        print(f"Error fetching order status: {e}")
        return None


@tool
def get_order_items(order_id: int) -> List[Dict[str, Any]]:
    """Get the line items (products ordered) for a specific order."""
    print(f"TOOL: get_order_items")
    url = f"{_base_url()}/orders/{order_id}.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        order = response.json().get("order")
        if not order:
            return []
        return [
            {
                "title": item.get("title"),
                "variant_title": item.get("variant_title"),
                "quantity": item.get("quantity"),
                "price": item.get("price"),
                "sku": item.get("sku"),
            }
            for item in order.get("line_items", [])
        ]
    except Exception as e:
        print(f"Error fetching order items: {e}")
        return []


@tool
def get_order_tracking(order_id: int) -> List[Dict[str, Any]]:
    """Get shipment tracking information for a specific order."""
    print(f"TOOL: get_order_tracking")
    url = f"{_base_url()}/orders/{order_id}/fulfillments.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        fulfillments = response.json().get("fulfillments", [])
        return [
            {
                "status": f.get("status"),
                "tracking_company": f.get("tracking_company"),
                "tracking_number": f.get("tracking_number"),
                "tracking_url": f.get("tracking_url"),
                "created_at": f.get("created_at"),
                "updated_at": f.get("updated_at"),
            }
            for f in fulfillments
        ]
    except Exception as e:
        print(f"Error fetching order tracking: {e}")
        return []


@tool
def cancel_order(order_id: int, reason: str = "customer") -> Dict[str, Any]:
    """
    Cancel an order on behalf of a customer.
    Reason must be one of: customer, inventory, fraud, declined, other.
    """
    print(f"TOOL: cancel_order")
    url = f"{_base_url()}/orders/{order_id}/cancel.json"
    try:
        response = requests.post(
            url,
            headers=_shopify_headers(),
            json={"reason": reason},
        )
        response.raise_for_status()
        order = response.json().get("order", {})
        return {
            "order_id": order.get("id"),
            "cancelled_at": order.get("cancelled_at"),
            "cancel_reason": order.get("cancel_reason"),
            "financial_status": order.get("financial_status"),
        }
    except Exception as e:
        print(f"Error cancelling order: {e}")
        return {"error": str(e)}


@tool
def get_cancellation_policy() -> str:
    """Fetch the store's cancellation / refund policy from Shopify."""
    url = f"{_base_url()}/policies.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        policies = response.json().get("policies", [])
        for policy in policies:
            if "refund" in policy.get("title", "").lower():
                return policy.get("body", "").strip()
        # fall back to first policy if no explicit refund policy
        return policies[0].get("body", "").strip() if policies else "No cancellation policy found."
    except Exception as e:
        return f"Unable to fetch cancellation policy: {e}"


ORDERS_TOOL_FUNCTIONS = [
    get_orders_by_email,
    get_order_by_id,
    get_order_status,
    get_order_items,
    get_order_tracking,
    cancel_order,
    get_cancellation_policy,
]
