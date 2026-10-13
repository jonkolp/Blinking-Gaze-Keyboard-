"""
Virtual On-Screen Keyboard module for eye-gaze typing.

Provides the VirtualKeyboard class that manages layout geometry,
collision detection with gaze coordinates, visual rendering with OpenCV,
and keystroke buffer manipulation (character appending, space, backspace, clear).
"""

from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np


class Key:
    """Represents an individual interactive key with its bounding box and label."""

    def __init__(self, label: str, x: int, y: int, w: int, h: int):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if a 2D coordinate falls within this key's bounding box."""
        px, py = point
        return self.x <= px <= (self.x + self.w) and self.y <= py <= (self.y + self.h)


class VirtualKeyboard:
    """
    Renders an interactive virtual keyboard canvas and processes gaze-based selections.
    """

    def __init__(self, key_size: int = 90, margin: int = 10, header_height: int = 110):
        self.key_size = key_size
        self.margin = margin
        self.header_height = header_height
        self.text = ""
        self.selected_key: Optional[str] = None
        self.last_pressed_key: Optional[str] = None

        # Build keyboard layout geometry
        self.keys: List[Key] = []
        self._build_layout()

    def _build_layout(self):
        """Build the geometric bounding boxes for all keyboard keys."""
        # Row 0: 10 keys
        row0 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
        # Row 1: 9 keys (staggered)
        row1 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
        # Row 2: 7 keys (staggered)
        row2 = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']

        self.width = 10 * self.key_size + 11 * self.margin

        # Populate Alphabet rows
        current_y = self.header_height + self.margin

        # Row 0
        for i, key in enumerate(row0):
            x = self.margin + i * (self.key_size + self.margin)
            self.keys.append(Key(key, x, current_y, self.key_size, self.key_size))

        # Row 1 (indented)
        current_y += self.key_size + self.margin
        row1_offset = self.margin + self.key_size // 2
        for i, key in enumerate(row1):
            x = row1_offset + i * (self.key_size + self.margin)
            self.keys.append(Key(key, x, current_y, self.key_size, self.key_size))

        # Row 2 (indented)
        current_y += self.key_size + self.margin
        row2_offset = self.margin + int(self.key_size * 1.5)
        for i, key in enumerate(row2):
            x = row2_offset + i * (self.key_size + self.margin)
            self.keys.append(Key(key, x, current_y, self.key_size, self.key_size))

        # Row 3: Action keys (SPACE, BACKSPACE, CLEAR)
        current_y += self.key_size + self.margin
        action_h = int(self.key_size * 0.75)
        space_w = self.key_size * 4 + self.margin * 3
        backspace_w = int(self.key_size * 2.2)
        clear_w = int(self.key_size * 1.8)

        # Center action row
        total_action_w = space_w + backspace_w + clear_w + 2 * self.margin
        action_start_x = max(self.margin, (self.width - total_action_w) // 2)

        x_cursor = action_start_x
        self.keys.append(Key("SPACE", x_cursor, current_y, space_w, action_h))
        x_cursor += space_w + self.margin
        self.keys.append(Key("BACKSPACE", x_cursor, current_y, backspace_w, action_h))
        x_cursor += backspace_w + self.margin
        self.keys.append(Key("CLEAR", x_cursor, current_y, clear_w, action_h))

        self.height = current_y + action_h + self.margin

    def detect_key(self, gaze_point: Optional[Tuple[int, int]]) -> Optional[str]:
        """
        Identify which key is currently targeted by the given gaze coordinate.

        :param gaze_point: (x, y) coordinates of the gaze point.
        :return: Label of targeted key or None.
        """
        if gaze_point is None:
            self.selected_key = None
            return None

        for key in self.keys:
            if key.contains(gaze_point):
                self.selected_key = key.label
                return key.label

        self.selected_key = None
        return None

    def press_key(self, key_label: str) -> str:
        """
        Process a keystroke action and update internal text buffer.

        :param key_label: Label of key pressed.
        :return: Updated text string.
        """
        self.last_pressed_key = key_label
        if key_label == "SPACE":
            self.text += " "
        elif key_label == "BACKSPACE":
            self.text = self.text[:-1]
        elif key_label == "CLEAR":
            self.text = ""
        elif len(key_label) == 1:
            self.text += key_label

        return self.text

    def render(self) -> np.ndarray:
        """
        Render the virtual keyboard canvas with keys, highlight states, and text banner.

        :return: BGR image representing the keyboard.
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (35, 35, 35)  # Dark background

        # Draw Top Header Text Banner
        cv2.rectangle(canvas, (self.margin, self.margin),
                      (self.width - self.margin, self.header_height - self.margin),
                      (50, 50, 50), -1)
        cv2.rectangle(canvas, (self.margin, self.margin),
                      (self.width - self.margin, self.header_height - self.margin),
                      (100, 100, 100), 2)

        # Render current typed string
        display_str = f"Typed: {self.text}_"
        cv2.putText(canvas, display_str, (self.margin + 20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Helper hint
        hint = "Gaze at a key & Blink to type  |  Keys: [SPACE] [BACKSPACE] [CLEAR]"
        cv2.putText(canvas, hint, (self.margin + 20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Draw Keys
        for key in self.keys:
            is_selected = (key.label == self.selected_key)

            if is_selected:
                # Highlighted state: bright teal fill
                bg_color = (180, 240, 200)
                border_color = (0, 255, 0)
                text_color = (0, 0, 0)
            else:
                # Default state: clean light key
                bg_color = (235, 235, 235)
                border_color = (180, 180, 180)
                text_color = (20, 20, 20)

            # Key rectangle
            cv2.rectangle(canvas, (key.x, key.y), (key.x + key.w, key.y + key.h), bg_color, -1)
            cv2.rectangle(canvas, (key.x, key.y), (key.x + key.w, key.y + key.h), border_color, 2)

            # Text centering
            font_scale = 0.65 if len(key.label) > 1 else 0.9
            thickness = 2
            (tw, th), _ = cv2.getTextSize(key.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tx = key.x + (key.w - tw) // 2
            ty = key.y + (key.h + th) // 2
            cv2.putText(canvas, key.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

            # Draw red focus indicator dot if selected
            if is_selected:
                center = (key.x + key.w // 2, key.y + key.h - 15)
                cv2.circle(canvas, center, 6, (0, 0, 255), -1)

        return canvas
