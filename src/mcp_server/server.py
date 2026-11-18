"""BIST harness MCP sunucusu (mcp 2.x MCPServer, eski adıyla FastMCP). 15 deterministik araç.

Çalıştırma (stdio):  python -m src.mcp_server.server
Araç gövdeleri src/mcp_server/tools.py içinde; burada yalnızca kaydedilir. Hiçbir araç emir göndermez,
broker'a bağlanmaz ya da veri yazmaz (yalnızca okuma).
"""
from mcp.server.mcpserver import MCPServer

from src.mcp_server.tools import TOOLS

INSTRUCTIONS = (
    "BIST 100 eğitim/araştırma araçları. Tüm sayılar deterministik Python hesaplarından gelir; "
    "araç çıktısında olmayan sayı uydurulmaz. 'asof' sonrası veri döndürülmez. "
    "Sonuçlar yatırım tavsiyesi değildir; son kararı insan verir."
)


def build_server() -> MCPServer:
    server = MCPServer("bist-harness", instructions=INSTRUCTIONS, version="1.0")
    for fn in TOOLS:
        server.add_tool(fn, name=fn.__name__, description=(fn.__doc__ or "").strip())
    return server


server = build_server()

if __name__ == "__main__":
    server.run("stdio")
