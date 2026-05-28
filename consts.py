"""Shared UI style constants for the Mesh Reviewer."""

from __future__ import annotations

SCENE_BG = "#181825"

HEADER = "bg-[#1e1e2e] border-b border-gray-700 px-6 py-3 items-center justify-between"
FOOTER = "bg-[#1e1e2e] border-t border-gray-700 px-6 py-2"
PANEL = "w-full h-full p-3 gap-2 bg-[#1e1e2e]"
CONTENT_AREA = "w-full h-[calc(100vh-110px)]"
SECTION_HEADER = "text-xs font-semibold text-gray-400 uppercase tracking-wider"
BADGE_OK = "items-center gap-1 bg-white/5 rounded-full px-3 py-1"


def inject_styles(ui_module) -> None:
    """Add global CSS overrides for the dark theme."""
    ui_module.add_css(
        """
        body {
            background-color: #11111b;
            color: #cdd6f4;
        }
        .q-splitter__separator {
            background-color: #313244 !important;
        }
        """
    )
