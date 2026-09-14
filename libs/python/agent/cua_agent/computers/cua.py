"""
Computer handler implementation for OpenAI computer-use-preview protocol.
"""

import base64
from typing import Any, Dict, List, Literal, Optional, Union

from computer import Computer

from .base import AsyncComputerHandler


class cuaComputerHandler(AsyncComputerHandler):
    """Computer handler that implements the Computer protocol using the computer interface."""

    def __init__(self, cua_computer: Computer):
        """Initialize with a computer interface (from tool schema)."""
        self.cua_computer = cua_computer
        self.interface = None

    async def _initialize(self):
        if hasattr(self.cua_computer, "_initialized") and not self.cua_computer._initialized:
            await self.cua_computer.run()
        self.interface = self.cua_computer.interface

    # ==== Computer-Use-Preview Action Space ====

    async def get_environment(self) -> Literal["windows", "mac", "linux", "browser"]:
        """Get the current environment type."""
        # TODO: detect actual environment
        return "linux"

    async def get_dimensions(self) -> tuple[int, int]:
        """Get screen dimensions as (width, height)."""
        assert self.interface is not None
        screen_size = await self.interface.get_screen_size()
        return screen_size["width"], screen_size["height"]

    async def screenshot(self, text: Optional[str] = None) -> str:
        """Take a screenshot and return as base64 string.

        Args:
            text: Optional descriptive text (for compatibility with GPT-4o models, ignored)
        """
        assert self.interface is not None
        screenshot_bytes = await self.interface.screenshot()
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def click(self, x: int, y: int, button: str = "left") -> None:
        """Click at coordinates with specified button."""
        assert self.interface is not None
        if button == "left":
            await self.interface.left_click(x, y)
        elif button == "right":
            await self.interface.right_click(x, y)
        else:
            # Default to left click for unknown buttons
            await self.interface.left_click(x, y)

    async def double_click(self, x: int, y: int) -> None:
        """Double click at coordinates."""
        assert self.interface is not None
        await self.interface.double_click(x, y)

    async def right_click(self, x: int, y: int) -> None:
        """Right click at coordinates."""
        assert self.interface is not None
        await self.interface.right_click(x, y)

    # Roughly one wheel detent per this many pixels of requested scroll.
    _SCROLL_PIXELS_PER_CLICK = 50
    # At or below this magnitude, scroll_y is already a number of detents.
    _SCROLL_DETENT_MAX = 25
    # Never emit a scroll too small to move the page, or large enough to run away.
    _SCROLL_MIN_CLICKS = 3
    _SCROLL_MAX_CLICKS = 30

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at coordinates with specified scroll amounts.

        `interface.scroll(x, y)` forwards its two arguments to the
        computer-server as {"x": ..., "y": ...}, which that endpoint does not
        act on: it answers {"success": true} while the page stays put. Only the
        detent-based scroll_down / scroll_up commands actually scroll.

        scroll_y also arrives in two incompatible units -- models emit both a
        small number of wheel notches (3-5) and pixel-like amounts (300-1000) --
        so small magnitudes are read as detents and larger ones as pixels.
        Positive scroll_y means scroll down, matching the tool schema.
        """
        assert self.interface is not None
        await self.interface.move_cursor(x, y)
        if not scroll_y:
            return

        magnitude = abs(scroll_y)
        if magnitude <= self._SCROLL_DETENT_MAX:
            clicks = magnitude
        else:
            clicks = round(magnitude / self._SCROLL_PIXELS_PER_CLICK)
        clicks = max(self._SCROLL_MIN_CLICKS, min(self._SCROLL_MAX_CLICKS, clicks))

        if scroll_y > 0:
            await self.interface.scroll_down(clicks)
        else:
            await self.interface.scroll_up(clicks)

    async def type(self, text: str) -> None:
        """Type text."""
        assert self.interface is not None
        await self.interface.type_text(text)

    async def wait(self, ms: int = 1000) -> None:
        """Wait for specified milliseconds."""
        assert self.interface is not None
        import asyncio

        await asyncio.sleep(ms / 1000.0)

    async def move(self, x: int, y: int) -> None:
        """Move cursor to coordinates."""
        assert self.interface is not None
        await self.interface.move_cursor(x, y)

    async def keypress(self, keys: Union[List[str], str]) -> None:
        """Press key combination."""
        assert self.interface is not None
        if isinstance(keys, str):
            keys = keys.replace("-", "+").split("+")
        if len(keys) == 1:
            await self.interface.press_key(keys[0])
        else:
            # Handle key combinations
            await self.interface.hotkey(*keys)

    async def drag(
        self,
        path: Optional[List[Dict[str, int]]] = None,
        start_x: Optional[int] = None,
        start_y: Optional[int] = None,
        end_x: Optional[int] = None,
        end_y: Optional[int] = None,
    ) -> None:
        """Drag along specified path or from start to end coordinates.

        Supports two formats:
        - path: List of {x, y} points to drag through
        - start_x, start_y, end_x, end_y: Simple drag from start to end
        """
        assert self.interface is not None

        # If start/end coordinates provided, convert to path format
        if start_x is not None and start_y is not None and end_x is not None and end_y is not None:
            path = [{"x": start_x, "y": start_y}, {"x": end_x, "y": end_y}]

        if not path:
            return

        # Start drag from first point
        start = path[0]
        await self.interface.mouse_down(start["x"], start["y"])

        # Move through path
        for point in path[1:]:
            await self.interface.move_cursor(point["x"], point["y"])

        # End drag at last point
        end = path[-1]
        await self.interface.mouse_up(end["x"], end["y"])

    async def terminate(self, status: str = "success") -> Dict[str, Any]:
        """Terminate the current task and report its completion status.

        Args:
            status: Status of the task ("success" or "failure")

        Returns:
            Dict with terminated flag and status
        """
        return {"success": True, "status": status, "terminated": True}

    async def get_current_url(self) -> str:
        """Get current URL (for browser environments)."""
        # This would need to be implemented based on the specific browser interface
        # For now, return empty string
        return ""

    # ==== Anthropic Computer Action Space ====
    async def left_mouse_down(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Left mouse down at coordinates."""
        assert self.interface is not None
        await self.interface.mouse_down(x, y, button="left")

    async def left_mouse_up(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Left mouse up at coordinates."""
        assert self.interface is not None
        await self.interface.mouse_up(x, y, button="left")

    # ==== Browser Control Methods (via Playwright) ====
    async def playwright_exec(
        self, command: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a Playwright browser command.

        Supports: visit_url, click, type, scroll, web_search, screenshot,
                  get_current_url, go_back, go_forward

        Args:
            command: The browser command to execute
            params: Command parameters

        Returns:
            Dict containing the command result
        """
        assert self.interface is not None
        return await self.interface.playwright_exec(command, params or {})
