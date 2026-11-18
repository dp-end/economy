"""MCP istemci demosu: sunucuyu ayrı süreç olarak (stdio) başlatır, araç listesini alır, her aracı bir kez çağırır
ve geçersiz bir girişin çökme yerine hata mesajı verdiğini gösterir.

Çalıştırma:  python -m src.mcp_server.client_demo
"""
import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[2]
DEMO_CALLS = {
    "get_stock_history": dict(ticker="ASELS", start="2026-11-02", end="2026-11-06"),
    "get_index_history": dict(start="2026-11-02"),
    "calculate_indicators": dict(ticker="ASELS", start="2026-11-04", indicators=["rsi_14", "st_dir", "kama_slope"]),
    "detect_reversal_events": dict(ticker="PETKM", start="2026-10-20"),
    "find_support_resistance": dict(ticker="ASELS", date="2026-11-06"),
    "rank_sector_relative_strength": dict(sector="Banking", horizon=10, date="2026-11-06"),
    "test_weekday_pattern": dict(ticker="PORTFOLIO", pattern="weekday"),
    "get_fundamental_metrics": dict(ticker="ASELS", asof="2026-11-06"),
    "calculate_fundamental_change": dict(ticker="TURSG", periods=3, asof="2026-11-06"),
    "get_macro_context": dict(start="2026-11-04", end="2026-11-06"),
    "get_news_events": dict(ticker="TURSG", start="2026-10-15", asof="2026-11-06"),
    "get_speech_features": dict(source_id="fed_20261028"),
    "run_backtest": dict(strategy="E_haber", start="2025-11-06"),
    "check_data_quality": dict(ticker="TURSG", asof="2026-11-06"),
    "build_evidence_bundle": dict(ticker="ASELS", asof="2026-11-06"),
}


def text_of(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server.server"], cwd=str(ROOT))
    async with Client(params, read_timeout_seconds=300) as client:
        tools = (await client.list_tools()).tools
        print(f"araç sayısı: {len(tools)}")
        for t in tools:
            print(f"  - {t.name}: {(t.description or '').splitlines()[0][:90]}")
        print("\nHer araç bir kez:")
        for name, args in DEMO_CALLS.items():
            t0 = time.time()
            r = await client.call_tool(name, args)
            body = text_of(r)
            status = "HATA" if r.is_error else "ok"
            print(f"  {name:31s} {status:4s} {len(body):6d} bayt {time.time() - t0:5.1f}s | {body[:80].replace(chr(10), ' ')}")
        print("\nGeçersiz girişler (çökme yerine hata mesajı):")
        for name, args in [("get_stock_history", dict(ticker="YOKBOYLE")), ("run_backtest", dict(strategy="AL_SAT")),
                           ("get_fundamental_metrics", dict(ticker="ASELS", asof="2030-01-01"))]:
            r = await client.call_tool(name, args)
            print(f"  {name}: is_error={r.is_error} | {text_of(r)[:110]}")
        r = await client.call_tool("build_evidence_bundle", DEMO_CALLS["build_evidence_bundle"])
        b = json.loads(text_of(r))
        print(f"\nASELS 2026-11-06 -> {b['action']} (kapı {b['gate']['status']}), {len(b['committed'])} kanıt")


if __name__ == "__main__":
    asyncio.run(main())
