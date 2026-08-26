"""Tests for Opower."""

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import aiohttp
import pytest

from opower import (
    Account,
    AggregateType,
    MeterType,
    Opower,
    ReadResolution,
    create_cookie_jar,
    get_supported_utilities,
)
from opower.exceptions import ApiException, InvalidAuth
from opower.opower import Customer

if TYPE_CHECKING:
    from opower.utilities import UtilityBase


@pytest.mark.parametrize("utility", get_supported_utilities())
@pytest.mark.asyncio
async def test_invalid_auth(utility: type["UtilityBase"]) -> None:
    """Test invalid username/password raises InvalidAuth."""
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            utility.name(),
            username="test",
            password="test",  # noqa: S106
            optional_totp_secret=None,
        )
        with pytest.raises(InvalidAuth):
            await opower.async_login()


@pytest.mark.asyncio
async def test_cost_reads_falls_back_to_usage_on_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cost endpoint errors fall back to usage-only reads for non-bill aggregations."""
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            "Pacific Gas and Electric Company (PG&E)",
            username="test",
            password="test",  # noqa: S106
        )

        account = Account(
            customer=Mock(),
            uuid="test-uuid",
            utility_account_id="test-id",
            id="test-id",
            meter_type=MeterType.ELEC,
            read_resolution=ReadResolution.HOUR,
        )

        call_log: list[bool] = []  # tracks usage_only values

        async def fake_get_dated_data(
            acc: object,
            agg: AggregateType,
            start: object,
            end: object,
            usage_only: bool = False,
        ) -> list[dict[str, object]]:
            call_log.append(usage_only)
            if not usage_only:
                raise ApiException(message="HTTP Error: 500", url="http://example.com")
            return [
                {
                    "startTime": "2026-01-01T00:00:00-05:00",
                    "endTime": "2026-01-02T00:00:00-05:00",
                    "consumption": {"value": 10.0},
                }
            ]

        monkeypatch.setattr(opower, "_async_get_dated_data", fake_get_dated_data)

        result = await opower.async_get_cost_reads(account, AggregateType.DAY, None, None)
        # Should have tried cost first, then fallen back to usage-only
        assert call_log == [False, True]
        assert len(result) == 1
        assert result[0].consumption == 10.0
        assert result[0].provided_cost == 0.0


@pytest.mark.asyncio
async def test_cost_reads_bill_raises_when_graphql_fallback_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing bill cost endpoint re-raises when the GraphQL fallback yields nothing."""
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            "Pacific Gas and Electric Company (PG&E)",
            username="test",
            password="test",  # noqa: S106
        )

        account = Account(
            customer=Customer(uuid="customer-uuid"),
            uuid="test-uuid",
            utility_account_id="test-id",
            id="test-id",
            meter_type=MeterType.ELEC,
            read_resolution=ReadResolution.HOUR,
        )

        async def fake_get_dated_data(*args: object, **kwargs: object) -> list[object]:
            raise ApiException(message="HTTP Error: 500", url="http://example.com")

        async def fake_post_graphql(*args: object, **kwargs: object) -> object:
            raise ApiException(message="GraphQL Error", url="http://example.com")

        monkeypatch.setattr(opower, "_async_get_dated_data", fake_get_dated_data)
        monkeypatch.setattr(opower, "_async_post_graphql", fake_post_graphql)

        with pytest.raises(ApiException):
            await opower.async_get_cost_reads(account, AggregateType.BILL, None, None)


def _pse_bill(
    interval: str,
    elec_charge: float,
    elec_usage: float,
    gas_charge: float,
    gas_usage: float,
    elec_total_energy_cost: float | None = None,
) -> dict[str, object]:
    """Build a WDB_GetCostUsageReadsForBills bill with an electric and a gas segment."""

    def _segment(
        uuid: str, service_type: str, charge: float, usage: float, unit: str, total: float | None
    ) -> dict[str, object]:
        return {
            "estimated": False,
            "usageInterval": interval,
            "usageCharges": {"value": charge},
            "currentAmount": {"value": charge},
            "deferredNEMCharges": None,
            "totalNEMCharges": None,
            "energyPurchased": None,
            "energySold": None,
            "rolloverBalanceEarned": None,
            "rolloverBalanceUsed": None,
            "totalEnergyCosts": {"value": total} if total is not None else None,
            "serviceAgreement": {
                "urn": f"urn:opower:v0:utilityAccount:pse:uuid:{uuid}",
                "uuid": uuid,
                "serviceType": service_type,
            },
            "serviceQuantities": [
                {
                    "unit": unit,
                    "serviceQuantityIdentifier": "NET_USAGE",
                    "utilityServiceQuantityIdentifier": None,
                    "serviceQuantity": {"value": usage},
                }
            ],
        }

    return {
        "timeInterval": interval,
        "segments": [
            _segment("elec-uuid", "ELECTRICITY", elec_charge, elec_usage, "KWH", elec_total_energy_cost),
            _segment("gas-uuid", "GAS", gas_charge, gas_usage, "TH", None),
        ],
    }


@pytest.mark.asyncio
async def test_cost_reads_bill_falls_back_to_graphql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the cost endpoint returns nothing, bill cost comes from the WDB GraphQL query.

    Mirrors utilities like PSE where DataBrowser-v1 authenticates but never
    returns cost data. Only the segment matching the account's uuid is used, and
    NET_USAGE can be negative for a period that exports more than it imports.
    """
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            "Puget Sound Energy (PSE)",
            username="test",
            password="test",  # noqa: S106
        )

        account = Account(
            customer=Customer(uuid="customer-uuid"),
            uuid="elec-uuid",
            utility_account_id="test-id",
            id="test-id",
            meter_type=MeterType.ELEC,
            read_resolution=ReadResolution.BILLING,
        )

        async def fake_get_dated_data(*args: object, **kwargs: object) -> list[dict[str, object]]:
            return []

        graphql_calls: list[dict[str, object]] = []

        async def fake_post_graphql(
            query: str,
            headers: dict[str, str],
            variables: dict[str, object] | None = None,
        ) -> dict[str, object]:
            graphql_calls.append(variables or {})
            return {
                "data": {
                    "billingAccountByAuthContext": {
                        "urn": "urn:opower:v0:multiCustomer:pse:uuids:customer-uuid",
                        "bills": [
                            _pse_bill(
                                "2025-12-06T00:00:00-08:00/2026-01-06T00:00:00-08:00",
                                elec_charge=302.19,
                                elec_usage=1743.0,
                                gas_charge=60.45,
                                gas_usage=46.263,
                            ),
                            _pse_bill(
                                "2026-06-05T00:00:00-07:00/2026-07-08T00:00:00-07:00",
                                elec_charge=0.0,
                                elec_usage=-123.0,
                                gas_charge=32.31,
                                gas_usage=24.281,
                            ),
                        ],
                    }
                }
            }

        monkeypatch.setattr(opower, "_async_get_dated_data", fake_get_dated_data)
        monkeypatch.setattr(opower, "_async_post_graphql", fake_post_graphql)

        result = await opower.async_get_cost_reads(account, AggregateType.BILL, None, None)

        tz = ZoneInfo("America/Los_Angeles")
        assert len(result) == 2
        # Only the electric segment is kept; gas charges/usage are excluded.
        assert result[0].provided_cost == 302.19
        assert result[0].consumption == 1743.0
        assert result[0].start_time == datetime(2025, 12, 6, tzinfo=tz)
        assert result[0].end_time == datetime(2026, 1, 6, tzinfo=tz)
        # Zero cost with negative net usage is retained (maps to return-to-grid).
        assert result[1].provided_cost == 0.0
        assert result[1].consumption == -123.0
        # First attempt carries the explicit selectedAccount URN.
        assert graphql_calls[0]["selectedAccount"] == "urn:opower:v0:multiCustomer:pse:uuids:customer-uuid"


@pytest.mark.asyncio
async def test_cost_reads_bill_graphql_retries_without_selected_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the query fails with an explicit selectedAccount, it retries without one."""
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            "Puget Sound Energy (PSE)",
            username="test",
            password="test",  # noqa: S106
        )

        account = Account(
            customer=Customer(uuid="customer-uuid"),
            uuid="elec-uuid",
            utility_account_id="test-id",
            id="test-id",
            meter_type=MeterType.ELEC,
            read_resolution=ReadResolution.BILLING,
        )

        async def fake_get_dated_data(*args: object, **kwargs: object) -> list[dict[str, object]]:
            return []

        graphql_calls: list[dict[str, object]] = []

        async def fake_post_graphql(
            query: str,
            headers: dict[str, str],
            variables: dict[str, object] | None = None,
        ) -> dict[str, object]:
            graphql_calls.append(variables or {})
            if "selectedAccount" in (variables or {}):
                raise ApiException(message="GraphQL Error", url="http://example.com")
            bill = _pse_bill(
                "2025-12-06T00:00:00-08:00/2026-01-06T00:00:00-08:00",
                elec_charge=302.19,
                elec_usage=1743.0,
                gas_charge=60.45,
                gas_usage=46.263,
            )
            return {"data": {"billingAccountByAuthContext": {"bills": [bill]}}}

        monkeypatch.setattr(opower, "_async_get_dated_data", fake_get_dated_data)
        monkeypatch.setattr(opower, "_async_post_graphql", fake_post_graphql)

        result = await opower.async_get_cost_reads(account, AggregateType.BILL, None, None)

        assert len(graphql_calls) == 2
        assert "selectedAccount" in graphql_calls[0]
        assert "selectedAccount" not in graphql_calls[1]
        assert len(result) == 1
        assert result[0].provided_cost == 302.19


@pytest.mark.asyncio
async def test_cost_reads_bill_graphql_prefers_total_energy_costs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NEM-aware totalEnergyCosts field wins over usageCharges when the utility populates it."""
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            "Puget Sound Energy (PSE)",
            username="test",
            password="test",  # noqa: S106
        )

        account = Account(
            customer=Customer(uuid="customer-uuid"),
            uuid="elec-uuid",
            utility_account_id="test-id",
            id="test-id",
            meter_type=MeterType.ELEC,
            read_resolution=ReadResolution.BILLING,
        )

        async def fake_get_dated_data(*args: object, **kwargs: object) -> list[dict[str, object]]:
            return []

        async def fake_post_graphql(
            query: str,
            headers: dict[str, str],
            variables: dict[str, object] | None = None,
        ) -> dict[str, object]:
            bill = _pse_bill(
                "2025-12-06T00:00:00-08:00/2026-01-06T00:00:00-08:00",
                elec_charge=120.0,
                elec_usage=800.0,
                gas_charge=40.0,
                gas_usage=30.0,
                elec_total_energy_cost=-15.5,
            )
            return {"data": {"billingAccountByAuthContext": {"bills": [bill]}}}

        monkeypatch.setattr(opower, "_async_get_dated_data", fake_get_dated_data)
        monkeypatch.setattr(opower, "_async_post_graphql", fake_post_graphql)

        result = await opower.async_get_cost_reads(account, AggregateType.BILL, None, None)

        assert len(result) == 1
        assert result[0].provided_cost == -15.5
        assert result[0].consumption == 800.0


@pytest.mark.asyncio
async def test_cost_reads_parse_read_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse readComponents (TOU/tier breakdown) when present, default to empty otherwise."""
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            "Sacramento Municipal Utility District (SMUD)",
            username="test",
            password="test",  # noqa: S106
        )

        account = Account(
            customer=Mock(),
            uuid="test-uuid",
            utility_account_id="test-id",
            id="test-id",
            meter_type=MeterType.ELEC,
            read_resolution=ReadResolution.HOUR,
        )

        async def fake_get_dated_data(*args: object, **kwargs: object) -> list[dict[str, object]]:
            return [
                # Real (redacted) SMUD read on a time-of-use rate.
                {
                    "startTime": "2026-06-17T00:00:00.000-07:00",
                    "endTime": "2026-06-18T00:00:00.000-07:00",
                    "value": 33.732,
                    "readType": "ACTUAL",
                    "providedCost": 6.9044064,
                    "readComponents": [
                        {
                            "tierType": "ORDINAL",
                            "tierNumber": None,
                            "season": "SUMMER",
                            "dayPart": "ON_PEAK+RT02/TOD",
                            "cost": 0.8195652,
                            "value": 2.1768,
                        },
                        {
                            "tierType": "ORDINAL",
                            "tierNumber": None,
                            "season": "SUMMER",
                            "dayPart": "OFF_PEAK+RT02/TOD",
                            "cost": 1.749516,
                            "value": 11.2872,
                        },
                        {
                            "tierType": "ORDINAL",
                            "tierNumber": None,
                            "season": "SUMMER",
                            "dayPart": "PART_PEAK+RT02/TOD",
                            "cost": 4.3353252,
                            "value": 20.268,
                        },
                    ],
                },
                # Read without components (not all utilities return them).
                {
                    "startTime": "2026-06-18T00:00:00.000-07:00",
                    "endTime": "2026-06-19T00:00:00.000-07:00",
                    "value": 31.104,
                    "readType": "ACTUAL",
                    "providedCost": 6.4527,
                },
            ]

        monkeypatch.setattr(opower, "_async_get_dated_data", fake_get_dated_data)

        result = await opower.async_get_cost_reads(account, AggregateType.DAY, None, None)
        assert len(result) == 2

        components = result[0].read_components
        assert len(components) == 3
        assert components[0].tier_type == "ORDINAL"
        assert components[0].tier_number is None
        assert components[0].season == "SUMMER"
        assert components[0].day_part == "ON_PEAK+RT02/TOD"
        assert components[0].cost == 0.8195652
        assert components[0].consumption == 2.1768
        # Components sum to the read's totals.
        assert sum(c.consumption for c in components) == pytest.approx(result[0].consumption)
        assert sum(c.cost for c in components) == pytest.approx(result[0].provided_cost)

        assert result[1].read_components == []


@pytest.mark.asyncio
async def test_five_minute_read_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse five-minute accounts and allow existing fine-grained aggregations."""
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            "Consolidated Edison (ConEd)",
            username="test",
            password="test",  # noqa: S106
        )

        async def fake_get_customers() -> list[dict[str, object]]:
            return [
                {
                    "uuid": "customer-uuid",
                    "utilityAccounts": [
                        {
                            "uuid": "account-uuid",
                            "preferredUtilityAccountId": "account-id",
                            "meterType": "ELEC",
                            "readResolution": "FIVE_MINUTE",
                        }
                    ],
                }
            ]

        calls: list[AggregateType] = []

        async def fake_async_fetch(
            account: Account,
            aggregate_type: AggregateType,
            start_date: datetime | None = None,
            end_date: datetime | None = None,
            usage_only: bool = False,
        ) -> list[dict[str, object]]:
            calls.append(aggregate_type)
            return [{"start": start_date, "end": end_date, "usage_only": usage_only}]

        monkeypatch.setattr(opower, "_async_get_customers", fake_get_customers)
        monkeypatch.setattr(opower, "_async_fetch", fake_async_fetch)

        accounts = await opower.async_get_accounts()

        assert len(accounts) == 1
        assert accounts[0].read_resolution is ReadResolution.FIVE_MINUTE
        await opower._async_get_dated_data(
            accounts[0],
            AggregateType.QUARTER_HOUR,
            datetime(2026, 1, 1),
            datetime(2026, 1, 1),
        )
        assert calls == [AggregateType.QUARTER_HOUR]


@pytest.mark.asyncio
async def test_naive_read_times_localized_to_utility_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reads without a UTC offset are localized to the utility's timezone.

    Some utilities (e.g. City of Austin) return timestamps with no offset.
    Statistics consumers (such as Home Assistant's recorder) require
    timezone-aware datetimes, so naive values must be localized rather than
    passed through unchanged.
    """
    async with aiohttp.ClientSession(cookie_jar=create_cookie_jar()) as session:
        opower = Opower(
            session,
            "City of Austin Utilities",
            username="test",
            password="test",  # noqa: S106
        )

        account = Account(
            customer=Mock(),
            uuid="test-uuid",
            utility_account_id="test-id",
            id="test-id",
            meter_type=MeterType.ELEC,
            read_resolution=ReadResolution.DAY,
        )

        async def fake_get_dated_data(
            *args: object,
            **kwargs: object,
        ) -> list[dict[str, object]]:
            return [
                {
                    "startTime": "2026-06-01T00:00:00",
                    "endTime": "2026-06-02T00:00:00",
                    "consumption": {"value": 10.0},
                    "providedCost": 1.23,
                }
            ]

        monkeypatch.setattr(opower, "_async_get_dated_data", fake_get_dated_data)

        result = await opower.async_get_cost_reads(account, AggregateType.DAY, None, None)

        tz = ZoneInfo("America/Chicago")
        assert len(result) == 1
        assert result[0].start_time == datetime(2026, 6, 1, tzinfo=tz)
        assert result[0].end_time == datetime(2026, 6, 2, tzinfo=tz)
        assert result[0].start_time.tzinfo is not None
        assert result[0].end_time.tzinfo is not None
