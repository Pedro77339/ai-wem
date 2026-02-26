"""Mock tools for testing the WEM engine without real backends."""

import json
import random
from datetime import datetime, timedelta

from ai_wem import ToolExecutor, ToolCall, ToolResult


# ── Mock tool definitions (OpenAI format) ─────────────────

def _tool(name, description, properties, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


MOCK_TOOLS = [
    _tool("get_weather", "Get current weather for a city", {
        "city": {"type": "string", "description": "City name"},
    }, required=["city"]),
    _tool("search_products", "Search products in the catalog", {
        "query": {"type": "string", "description": "Search keyword"},
        "category": {"type": "string", "description": "Product category (optional)"},
    }, required=["query"]),
    _tool("get_order_status", "Get status of an order by ID", {
        "order_id": {"type": "string", "description": "Order ID (e.g. ORD-1234)"},
    }, required=["order_id"]),
    _tool("calculate", "Evaluate a math expression", {
        "expression": {"type": "string", "description": "Math expression (e.g. '2+2', '100*0.15')"},
    }, required=["expression"]),
    _tool("get_system_info", "Get system information (mock)", {}),
]


# ── Worker classify prompt + intent map ───────────────────

CLASSIFY_PROMPT = (
    "Classify the user's intent into ONE category. "
    "Respond ONLY with JSON, no markdown:\n"
    '{{"intent": "category", "params": {{}}}}\n\n'
    "Categories:\n"
    "- weather: asking about the weather in a city (params: city=name)\n"
    "- products: searching products or catalog\n"
    "- order: order status (params: order_id=ID)\n"
    "- system: system info, status\n"
    "- none: doesn't fit any category\n\n"
    "Question: {user_text}"
)

INTENT_MAP = {
    "weather": ("get_weather", {"city": "New York"}),
    "products": ("search_products", {"query": "popular"}),
    "order": ("get_order_status", {"order_id": "ORD-0001"}),
    "system": ("get_system_info", {}),
}


# ── Mock executor ─────────────────────────────────────────

class MockToolExecutor(ToolExecutor):
    """Simulates tool execution with fake data."""

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        handlers = {
            "get_weather": self._weather,
            "search_products": self._products,
            "get_order_status": self._order,
            "calculate": self._calculate,
            "get_system_info": self._sysinfo,
        }
        handler = handlers.get(tool_call.name)
        if not handler:
            return ToolResult(tool_call.id, tool_call.name,
                              f"Unknown tool: {tool_call.name}", is_error=True)
        try:
            result = handler(**tool_call.arguments)
            return ToolResult(tool_call.id, tool_call.name, result)
        except Exception as ex:
            return ToolResult(tool_call.id, tool_call.name,
                              f"Error: {ex}", is_error=True)

    def _weather(self, city: str = "New York", **kw) -> str:
        temps = {"New York": 18, "London": 12, "Tokyo": 24, "Paris": 15}
        conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
        temp = temps.get(city, random.randint(15, 35))
        cond = random.choice(conditions)
        return json.dumps({
            "city": city, "temperature": temp, "condition": cond,
            "humidity": random.randint(40, 90), "wind_kph": random.randint(5, 25),
        })

    def _products(self, query: str = "", category: str = "", **kw) -> str:
        products = [
            {"id": 1, "name": "Water 600ml", "price": 2.50, "stock": 48},
            {"id": 2, "name": "Soda 350ml", "price": 3.00, "stock": 32},
            {"id": 3, "name": "Natural Juice", "price": 4.50, "stock": 15},
            {"id": 4, "name": "Cookie Pack", "price": 2.00, "stock": 60},
            {"id": 5, "name": "Chocolate Bar", "price": 1.50, "stock": 100},
        ]
        if query:
            q = query.lower()
            products = [p for p in products if q in p["name"].lower()] or products
        return json.dumps({"results": products, "total": len(products)})

    def _order(self, order_id: str = "ORD-0001", **kw) -> str:
        statuses = ["pending", "in preparation", "shipped", "delivered"]
        return json.dumps({
            "order_id": order_id,
            "status": random.choice(statuses),
            "items": 3,
            "total": round(random.uniform(10, 50), 2),
            "estimated_delivery": (datetime.now() + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d"),
        })

    def _calculate(self, expression: str = "0", **kw) -> str:
        # Safe eval — only math
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid expression"})
        try:
            result = eval(expression)
            return json.dumps({"expression": expression, "result": result})
        except Exception as ex:
            return json.dumps({"error": str(ex)})

    def _sysinfo(self, **kw) -> str:
        return json.dumps({
            "hostname": "demo-server",
            "os": "Linux 6.1",
            "cpu": "4 cores",
            "memory": "8GB (3.2GB used)",
            "uptime": "14 days",
            "python": "3.12.0",
        })
