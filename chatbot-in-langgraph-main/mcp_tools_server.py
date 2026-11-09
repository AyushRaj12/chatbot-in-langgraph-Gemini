from datetime import UTC, datetime
from typing import Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Public Read-Only Utilities",
    instructions="Deterministic read-only utilities. These tools do not access private data or external services.",
)


@mcp.tool()
def convert_temperature(
    value: float,
    from_unit: Literal["celsius", "fahrenheit", "kelvin"],
    to_unit: Literal["celsius", "fahrenheit", "kelvin"],
) -> dict[str, float | str]:
    """Convert a temperature between Celsius, Fahrenheit, and Kelvin."""
    if from_unit == "celsius":
        celsius = value
    elif from_unit == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    else:
        celsius = value - 273.15

    if celsius < -273.15:
        return {"error": "Temperature cannot be below absolute zero."}

    if to_unit == "celsius":
        result = celsius
    elif to_unit == "fahrenheit":
        result = celsius * 9 / 5 + 32
    else:
        result = celsius + 273.15

    return {
        "value": round(result, 4),
        "unit": to_unit,
        "source": "local read-only MCP utility",
    }


@mcp.tool()
def calculate_percentage(part: float, total: float) -> dict[str, float | str]:
    """Calculate what percentage one number is of another."""
    if total == 0:
        return {"error": "The total cannot be zero."}
    return {
        "percentage": round((part / total) * 100, 4),
        "source": "local read-only MCP utility",
    }


@mcp.tool()
def current_utc_time() -> dict[str, str]:
    """Return the current UTC date and time."""
    return {
        "utc_time": datetime.now(UTC).isoformat(),
        "source": "local read-only MCP utility",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
