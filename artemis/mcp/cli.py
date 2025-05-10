"""
ARTEMIS MCP Server CLI

Command-line interface for running the ARTEMIS MCP server.
"""

import argparse
import asyncio
import sys

from artemis.mcp.server import ArtemisMCPServer
from artemis.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="artemis-mcp",
        description="ARTEMIS MCP Server - Structured multi-agent debate via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start stdio server (for MCP clients)
  artemis-mcp

  # Start HTTP server
  artemis-mcp --http --port 8080

  # Use specific model
  artemis-mcp --model gpt-4-turbo

  # Enable verbose logging
  artemis-mcp --verbose
""",
    )

    parser.add_argument(
        "--http",
        action="store_true",
        help="Run as HTTP server instead of stdio",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Default LLM model for debates (default: gpt-4o)",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=100,
        help="Maximum concurrent debate sessions (default: 100)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser


async def run_server(args: argparse.Namespace) -> None:
    """Run the MCP server with given arguments."""
    server = ArtemisMCPServer(
        default_model=args.model,
        max_sessions=args.max_sessions,
    )

    if args.http:
        logger.info(
            "Starting ARTEMIS MCP HTTP server",
            host=args.host,
            port=args.port,
        )
        await server.start(host=args.host, port=args.port)
    else:
        logger.info("Starting ARTEMIS MCP stdio server")
        await server.run_stdio()


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    try:
        asyncio.run(run_server(args))
        return 0
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error("Server error", error=str(e))
        return 1


def run_cli() -> None:
    """Entry point for setuptools console_scripts."""
    sys.exit(main())


if __name__ == "__main__":
    run_cli()
