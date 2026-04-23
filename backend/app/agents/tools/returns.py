from typing import List, Dict, Optional, Any

import requests
from langchain_core.tools import tool

from app.settings import get_settings


def _shopify_headers() -> Dict[str, str]:
    return {
        "X-Shopify-Access-Token": get_settings().shopify_access_token,
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    return get_settings().shopify_base_url.rstrip("/")


def _graphql(query: str, variables: Optional[dict] = None) -> dict:
    """Execute a Shopify GraphQL Admin API request."""
    url = f"{_base_url()}/graphql.json"
    payload: dict = {"query": query}
    if variables:
        payload["variables"] = variables
    response = requests.post(url, headers=_shopify_headers(), json=payload)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# QnA / read-only tools (used by the free-form returns agent)
# ---------------------------------------------------------------------------

@tool
def get_returnable_orders_by_email(email: str) -> List[Dict[str, Any]]:
    """Fetch fulfilled orders for a customer email that are eligible for a return."""
    print("TOOL: get_returnable_orders_by_email")
    url = f"{_base_url()}/orders.json?email={email}&status=any&fulfillment_status=shipped"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        orders = response.json().get("orders", [])
        # Keep only orders that are fulfilled and not already fully refunded
        return [
            o for o in orders
            if o.get("fulfillment_status") == "fulfilled"
            and o.get("financial_status") not in ("refunded", "voided")
        ]
    except Exception as e:
        print(f"Error fetching returnable orders: {e}")
        return []


@tool
def get_return_policy() -> str:
    """Fetch the store's return / refund policy from Shopify."""
    print("TOOL: get_return_policy")
    url = f"{_base_url()}/policies.json"
    try:
        response = requests.get(url, headers=_shopify_headers())
        response.raise_for_status()
        policies = response.json().get("policies", [])
        for policy in policies:
            if "refund" in policy.get("title", "").lower():
                return policy.get("body", "").strip()
        return policies[0].get("body", "").strip() if policies else "No return policy found."
    except Exception as e:
        return f"Unable to fetch return policy: {e}"


@tool
def get_return_reasons() -> List[Dict[str, str]]:
    """Fetch the list of valid return reason definitions from Shopify."""
    print("TOOL: get_return_reasons")
    query = """
    {
      suggestedReturnReasonDefinitions {
        id
        handle
        name
      }
    }
    """
    try:
        data = _graphql(query)
        return data.get("data", {}).get("suggestedReturnReasonDefinitions", [])
    except Exception as e:
        print(f"Error fetching return reasons: {e}")
        return []


@tool
def get_existing_returns(order_id: int) -> List[Dict[str, Any]]:
    """Check if an order already has any return requests or completed returns."""
    print("TOOL: get_existing_returns")
    query = """
    query getOrderReturns($id: ID!) {
      order(id: $id) {
        returns(first: 10) {
          edges {
            node {
              id
              name
              status
              returnLineItems(first: 10) {
                edges {
                  node {
                    quantity
                    returnReason
                    returnReasonNote
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    gid = f"gid://shopify/Order/{order_id}"
    try:
        data = _graphql(query, {"id": gid})
        edges = (
            data.get("data", {})
            .get("order", {})
            .get("returns", {})
            .get("edges", [])
        )
        return [e["node"] for e in edges]
    except Exception as e:
        print(f"Error fetching existing returns: {e}")
        return []


# ---------------------------------------------------------------------------
# Workflow tools (used directly by the request_return workflow nodes)
# ---------------------------------------------------------------------------

def get_fulfillment_line_items(order_id: int) -> List[Dict[str, Any]]:
    """
    Internal helper (not a @tool) — returns fulfillment line items with their
    GraphQL IDs needed for the returnRequest mutation.
    """
    print("TOOL: get_fulfillment_line_items")
    query = """
    query getFulfillmentItems($id: ID!) {
      order(id: $id) {
        fulfillments(first: 10) {
          fulfillmentLineItems(first: 20) {
            edges {
              node {
                id
                quantity
                lineItem {
                  title
                  variantTitle
                  sku
                }
              }
            }
          }
        }
      }
    }
    """
    gid = f"gid://shopify/Order/{order_id}"
    try:
        data = _graphql(query, {"id": gid})
        items = []
        fulfillments = (
            data.get("data", {})
            .get("order", {})
            .get("fulfillments", [])
        )
        for fulfillment in fulfillments:
            for edge in fulfillment.get("fulfillmentLineItems", {}).get("edges", []):
                node = edge["node"]
                li = node.get("lineItem", {})
                items.append({
                    "fulfillment_line_item_id": node["id"],
                    "quantity": node["quantity"],
                    "title": li.get("title", "Unknown"),
                    "variant_title": li.get("variantTitle", ""),
                    "sku": li.get("sku", ""),
                })
        return items
    except Exception as e:
        print(f"Error fetching fulfillment line items: {e}")
        return []


def submit_return_request(
    order_id: int,
    fulfillment_line_item_id: str,
    quantity: int,
    reason_id: str,
    customer_note: str = "",
) -> Dict[str, Any]:
    """
    Internal helper (not a @tool) — submits a returnRequest GraphQL mutation.
    Returns the new return object or an error dict.
    """
    print("TOOL: submit_return_request")
    mutation = """
    mutation returnRequest($input: ReturnRequestInput!) {
      returnRequest(input: $input) {
        return {
          id
          name
          status
        }
        userErrors {
          field
          message
        }
      }
    }
    """
    variables = {
        "input": {
            "orderId": f"gid://shopify/Order/{order_id}",
            "returnLineItems": [
                {
                    "fulfillmentLineItemId": fulfillment_line_item_id,
                    "quantity": quantity,
                    "returnReasonDefinitionId": reason_id,
                    "customerNote": customer_note,
                }
            ],
        }
    }
    try:
        data = _graphql(mutation, variables)
        payload = data.get("data", {}).get("returnRequest", {})
        errors = payload.get("userErrors", [])
        if errors:
            return {"error": errors[0].get("message", "Unknown error")}
        return payload.get("return", {})
    except Exception as e:
        return {"error": str(e)}


RETURNS_TOOL_FUNCTIONS = [
    get_returnable_orders_by_email,
    get_return_policy,
    get_return_reasons,
    get_existing_returns,
]
