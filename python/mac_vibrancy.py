"""macOS window vibrancy (translucent "frosted glass" material) for Qt windows.

Inserts an ``NSVisualEffectView`` behind the Qt content so the desktop/other
windows show through with a live blur — the modern macOS materials look. The Qt
window is made translucent so the effect shows through wherever the Qt content
is itself semi-transparent (translucent cards / toolbars).

``apply_vibrancy(widget)`` is a best-effort no-op on non-macOS, without pyobjc,
or if anything goes wrong (it reverts the translucent attribute so the window
never ends up see-through-to-desktop by accident).
"""

from __future__ import annotations

import sys

# NSVisualEffectView material codes (AppKit enum). "Sidebar" is the light,
# lively frosted material used by Finder/Mail sidebars.
MATERIAL_SIDEBAR = 7
MATERIAL_HEADER = 10
MATERIAL_WINDOW_BACKGROUND = 12
MATERIAL_UNDER_WINDOW_BACKGROUND = 21

_NS_VIEW_WIDTH_SIZABLE = 2
_NS_VIEW_HEIGHT_SIZABLE = 16
_NS_WINDOW_BELOW = -1


def apply_vibrancy(widget, material: int = MATERIAL_SIDEBAR) -> bool:
    """Give ``widget`` (a top-level Qt window) a translucent macOS material
    background. Returns True on success, False otherwise."""
    if sys.platform != "darwin":
        return False
    try:
        import objc
        from AppKit import (NSVisualEffectView, NSVisualEffectBlendingModeBehindWindow,
                            NSVisualEffectStateActive)
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QGuiApplication
    except Exception:
        return False

    # Only touch a real Cocoa window. On "offscreen"/other platforms winId() is
    # not an NSView and calling Cocoa on it would segfault (uncatchable).
    if QGuiApplication.platformName() != "cocoa":
        return False

    try:
        widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        widget.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        view = objc.objc_object(c_void_p=int(widget.winId()))   # the Qt NSView
        window = view.window()
        if window is None:
            raise RuntimeError("no NSWindow yet")
        content = window.contentView()

        effect = NSVisualEffectView.alloc().initWithFrame_(content.bounds())
        effect.setAutoresizingMask_(_NS_VIEW_WIDTH_SIZABLE | _NS_VIEW_HEIGHT_SIZABLE)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        try:
            effect.setMaterial_(material)
        except Exception:
            pass
        # place it as the very bottom layer, behind Qt's content
        content.addSubview_positioned_relativeTo_(effect, _NS_WINDOW_BELOW, None)
        try:
            window.setTitlebarAppearsTransparent_(True)
        except Exception:
            pass
        return True
    except Exception as exc:                       # revert so we never leave a hole
        try:
            from PyQt6.QtCore import Qt
            widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            widget.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        except Exception:
            pass
        print(f"[vibrancy] not applied: {exc}")
        return False
