"""External API tools used by the research agent."""

import asyncio
from abc import ABC

import httpx

from agentflow.config import Settings
from agentflow.errors import ToolError
from agentflow.schemas import ToolResult


class ExternalApiTool(ABC):
    """Strategy object for one external HTTP API."""

    name = "tool"
    description = ""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    async def invoke(self, inputs: dict) -> ToolResult:
        raise NotImplementedError

    async def _get(self, url: str, params: dict | None = None) -> object:
        attempts = max(1, self.settings.http_retries)
        last_error: Exception | None = None
        headers = {"User-Agent": "AgentFlowEnterprise/0.1", "Accept": "application/json"}
        for attempt in range(1, attempts + 1):
            try:
                if self.client is not None:
                    response = await self.client.get(url, params=params, headers=headers)
                else:
                    async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
                        response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == attempts:
                    break
                await asyncio.sleep(self.settings.http_retry_delay * attempt)
        raise ToolError(f"{self.name} failed after {attempts} attempts: {last_error}")


class FxRatesTool(ExternalApiTool):
    """Latest rate from the Frankfurter exchange-rate API."""

    name = "fx_rates"
    description = "Latest foreign-exchange rate between two ISO currency codes."

    async def invoke(self, inputs: dict) -> ToolResult:
        base = str(inputs.get("base") or "USD").upper()
        quote = str(inputs.get("quote") or "EUR").upper()
        if base == quote:
            return ToolResult(
                name=self.name,
                ok=True,
                data={"base": base, "quote": quote, "rate": 1.0, "source": "identity"},
            )
        payload = await self._get("https://api.frankfurter.app/latest", {"from": base, "to": quote})
        rates = payload.get("rates", {}) if isinstance(payload, dict) else {}
        rate = rates.get(quote)
        if rate is None:
            return ToolResult(name=self.name, ok=False, error=f"No rate published for {base}/{quote}")
        return ToolResult(
            name=self.name,
            ok=True,
            data={
                "base": base,
                "quote": quote,
                "rate": rate,
                "date": payload.get("date") if isinstance(payload, dict) else None,
                "source": "frankfurter",
            },
        )


class WorldBankTool(ExternalApiTool):
    """Recent indicator observation from the World Bank API."""

    name = "world_bank"
    description = "Latest World Bank indicator value for a country."

    async def invoke(self, inputs: dict) -> ToolResult:
        country = str(inputs.get("country") or "US").upper()
        indicator = str(inputs.get("indicator") or "NY.GDP.MKTP.CD")
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        payload = await self._get(url, {"format": "json", "per_page": 8})
        if not isinstance(payload, list) or len(payload) < 2:
            message = ""
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                message = str(payload[0].get("message") or payload[0])
            return ToolResult(name=self.name, ok=False, error=message or "World Bank returned no observations")
        rows = payload[1] or []
        point = next((row for row in rows if isinstance(row, dict) and row.get("value") is not None), None)
        if point is None:
            return ToolResult(name=self.name, ok=False, error=f"No observation for {country} {indicator}")
        return ToolResult(
            name=self.name,
            ok=True,
            data={
                "country": country,
                "indicator": indicator,
                "indicator_name": (point.get("indicator") or {}).get("value"),
                "year": point.get("date"),
                "value": point.get("value"),
                "source": "world_bank",
            },
        )


class CountryInfoTool(ExternalApiTool):
    """Country profile from the REST Countries API."""

    name = "country_info"
    description = "Country name, region, capital, and currencies."

    async def invoke(self, inputs: dict) -> ToolResult:
        raw = str(inputs.get("country") or inputs.get("country_name") or "US").strip()
        if len(raw) <= 3 and raw.isalpha():
            url = f"https://restcountries.com/v3.1/alpha/{raw}"
        else:
            url = f"https://restcountries.com/v3.1/name/{raw}"
        payload = await self._get(
            url,
            {"fields": "name,region,subregion,currencies,capital,population,cca2"},
        )
        item = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(item, dict):
            return ToolResult(name=self.name, ok=False, error=f"No country profile for {raw}")
        name = item.get("name") or {}
        currencies = sorted((item.get("currencies") or {}).keys())
        return ToolResult(
            name=self.name,
            ok=True,
            data={
                "code": item.get("cca2") or raw.upper(),
                "name": name.get("common") if isinstance(name, dict) else str(name),
                "region": item.get("region"),
                "subregion": item.get("subregion"),
                "capital": (item.get("capital") or [None])[0],
                "currencies": currencies,
                "population": item.get("population"),
                "source": "rest_countries",
            },
        )


class ToolFactory:
    """Factory for the built-in external tools."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    def create_all(self) -> dict[str, ExternalApiTool]:
        tools = [
            FxRatesTool(self.settings, self.client),
            WorldBankTool(self.settings, self.client),
            CountryInfoTool(self.settings, self.client),
        ]
        return {tool.name: tool for tool in tools}
