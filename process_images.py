import os
from collections import Counter
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import numpy as np
import math
try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    import Tkinter as tk
    import tkMessageBox as messagebox
def get_background_color(image, default_color=(0, 0, 0, 0)):
    """
    Determines a suitable background color. First, it checks the corners for a
    uniform color. If that fails, it analyzes the colors along the border of
    the image content to find a suitable average color.

    Args:
        image (PIL.Image.Image): The image to analyze.
        default_color (tuple): The color to return if all analysis fails.

    Returns:
        tuple: An RGBA tuple for the calculated background color.
    """
    try:
        # Ensure image is RGBA for consistent pixel format
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        width, height = image.size
        # 1. Check corners for a predominant color (fast path for simple backgrounds)
        corners = [image.getpixel((0, 0)), image.getpixel((width - 1, 0)),
                   image.getpixel((0, height - 1)), image.getpixel((width - 1, height - 1))]
        
        # Use Counter to handle potential non-hashable list of lists in corners
        corner_counts = Counter(map(tuple, corners))
        most_common_corner, count = corner_counts.most_common(1)[0]

        if count >= 3: # If at least 3 corners match
            return most_common_corner

        # 2. If corners are not uniform, analyze the perimeter of the image
        # to find an average color that blends with the subject's edges.
        border_pixels = []
        # Top and bottom edges
        for x in range(width):
            border_pixels.append(image.getpixel((x, 0)))
            border_pixels.append(image.getpixel((x, height - 1)))
        # Left and right edges (excluding corners already added)
        for y in range(1, height - 1):
            border_pixels.append(image.getpixel((0, y)))
            border_pixels.append(image.getpixel((width - 1, y)))

        # Filter for non-transparent pixels to get the subject's edge colors
        opaque_border_pixels = [p for p in border_pixels if p[3] > 128]

        if opaque_border_pixels:
            # Calculate the average color of the opaque border pixels
            avg_r = sum(p[0] for p in opaque_border_pixels) // len(opaque_border_pixels)
            avg_g = sum(p[1] for p in opaque_border_pixels) // len(opaque_border_pixels)
            avg_b = sum(p[2] for p in opaque_border_pixels) // len(opaque_border_pixels)
            return (avg_r, avg_g, avg_b, 255)

    except (IndexError, ValueError):
        pass # Fall through to default

    return default_color

def make_square(image, method='blur'):
    """
    Converts the image to a square by extending the background.

    Args:
        image (PIL.Image.Image): Source image.
        method (str): 'blur' (fill with blurred original) or 'extend' (stretch edges).

    Returns:
        PIL.Image.Image: Squared image.
    """
    width, height = image.size
    if width == height:
        return image

    size = max(width, height)
    new_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))

    if method == 'blur':
        # Scale image to fill the square (maintain aspect ratio)
        ratio = size / min(width, height)
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        bg = image.resize((new_w, new_h), Image.LANCZOS)

        # Center crop the background
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        bg = bg.crop((left, top, left + size, top + size))

        # Blur the background
        bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
        new_img.paste(bg, (0, 0))

    elif method == 'extend':
        # Calculate position to paste original
        paste_x = (size - width) // 2
        paste_y = (size - height) // 2

        if height > width: # Portrait - extend sides
            # Left extension
            if paste_x > 0:
                left_col = image.crop((0, 0, 1, height))
                new_img.paste(left_col.resize((paste_x, height), Image.NEAREST), (0, paste_y))
            # Right extension
            if paste_x + width < size:
                right_col = image.crop((width - 1, 0, width, height))
                new_img.paste(right_col.resize((size - (paste_x + width), height), Image.NEAREST), (paste_x + width, paste_y))
        elif width > height: # Landscape - extend top/bottom
            # Top extension
            if paste_y > 0:
                top_row = image.crop((0, 0, width, 1))
                new_img.paste(top_row.resize((width, paste_y), Image.NEAREST), (paste_x, 0))
            # Bottom extension
            if paste_y + height < size:
                bottom_row = image.crop((0, height - 1, width, height))
                new_img.paste(bottom_row.resize((width, size - (paste_y + height)), Image.NEAREST), (paste_x, paste_y + height))

    # Paste original in center
    paste_x = (size - width) // 2
    paste_y = (size - height) // 2
    new_img.paste(image, (paste_x, paste_y), image if image.mode == 'RGBA' else None)

    return new_img

def get_content_radius(image):
    """
    Calculates the effective radius of the content in an image by finding the
    distance from the center to the furthest non-transparent pixel. This helps
    create a tighter circular crop for non-rectangular content.

    Args:
        image (PIL.Image.Image): The image to analyze (must be in RGBA mode).

    Returns:
        int: The calculated radius of the content.
    """
    width, height = image.size
    center_x, center_y = width / 2, height / 2

    # If the corners are transparent, it's likely non-rectangular content.
    # In this case, we scan for the 'true' radius.s
    
    corners_transparent = all(
        image.getpixel((x, y))[3] == 0
        for x in [0, width - 1] for y in [0, height - 1]
    )

    if corners_transparent:
        max_dist_sq = 0
        img_arr = np.array(image)
        non_transparent_pts = np.argwhere(img_arr[:, :, 3] > 0)
        if non_transparent_pts.size > 0:
            # Calculate squared distance from center for all non-transparent points
            distances_sq = ((non_transparent_pts[:, 1] - center_x) ** 2 +
                            (non_transparent_pts[:, 0] - center_y) ** 2)
            max_dist_sq = np.max(distances_sq)
        return math.ceil(math.sqrt(max_dist_sq)) if max_dist_sq > 0 else 0
    else:
        # For rectangular content, the radius must be half the diagonal
        # to ensure the corners are not clipped.
        return math.ceil(math.sqrt(width**2 + height**2) / 2)

class InteractiveImageEditor:
    def __init__(self, master, original_img, output_path, frame_radius, frame_width, bg_color, light_angle):
        self.master = master
        self.original_img = original_img
        self.output_path = output_path
        self.frame_radius = frame_radius
        self.frame_width = frame_width
        self.bg_color = bg_color
        self.light_angle = light_angle

        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_step = 10
        self.dragging = False
        self.last_mouse_x = None
        self.last_mouse_y = None
        self.left_dragged = False

        self.image_diameter = self.frame_radius * 2
        self.fill_layer = Image.new("RGBA", (self.image_diameter, self.image_diameter), (0, 0, 0, 0))
        self.fill_history = []
        self.patch_size = max(20, self.frame_radius // 8)
        self.sample_tile = None
        self.sample_tile_origin = (0, 0)
        self.selection_start_canvas = None
        self.selection_end_canvas = None
        self.selection_image_box = None
        self.selection_canvas_box = None
        self.sample_window = None
        self.sample_canvas = None
        self.space_pan = False

        # Determine a sensible canvas size that fits the screen
        max_dim = self.frame_radius * 2
        # Limit to 1/4 of screen real estate by using 50% of height
        screen_height = master.winfo_screenheight() 
        self.display_size = min(max_dim, int(screen_height * 0.5))
        
        self.canvas_size = self.display_size
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.start_left_action)
        self.canvas.bind("<B1-Motion>", self.do_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_left_action)
        self.canvas.bind("<ButtonPress-3>", self.handle_fill_click)

        # Add a frame for controls (Improved Layout)
        control_frame = tk.Frame(master, bd=1, relief=tk.RAISED)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        left_panel = tk.Frame(control_frame)
        left_panel.pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Button(left_panel, text="Save and Close", command=self.save_and_close, bg="#ddffdd").pack(side=tk.LEFT, padx=5)
        tk.Button(left_panel, text="Undo Fill (u)", command=self.undo_fill).pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(left_panel, text=" |  Left-drag: Select Tile  |  Click: Fill  |  Space+Drag: Pan")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Preview panel in main window
        right_panel = tk.Frame(control_frame)
        right_panel.pack(side=tk.RIGHT, padx=10, pady=5)
        tk.Label(right_panel, text="Tile:").pack(side=tk.LEFT)
        self.sample_canvas = tk.Canvas(right_panel, width=40, height=40, bg="#cccccc", bd=1, relief=tk.SUNKEN)
        self.sample_canvas.pack(side=tk.LEFT, padx=5)

        # Bind keys for interaction
        # Use 'z' for zoom in to avoid conflict with 'i' in input fields
        master.bind("<KeyPress-z>", lambda e: self.zoom_in(e, small_step=True))
        master.bind("<KeyPress-o>", lambda e: self.zoom_out(e, small_step=True))
        master.bind("<KeyPress-Z>", lambda e: self.zoom_in(e, small_step=False))
        master.bind("<KeyPress-O>", lambda e: self.zoom_out(e, small_step=False))
        master.bind("<KeyPress-Up>", self.pan_up)
        master.bind("<KeyPress-Down>", self.pan_down)
        master.bind("<KeyPress-Left>", self.pan_left)
        master.bind("<KeyPress-Right>", self.pan_right)
        master.bind("<KeyPress-u>", self.undo_fill)
        master.bind("<KeyPress-space>", self._space_down)
        master.bind("<KeyRelease-space>", self._space_up)

        # Pre-render the static frame overlay for performance
        self.frame_overlay = self._create_frame_overlay()

        self.redraw()

    def _create_frame_overlay(self):
        """Creates the frame as a static overlay with a transparent center."""
        image_diameter = self.frame_radius * 2
        center = self.frame_radius
        frame_inner_radius = self.frame_radius - self.frame_width
        gold_color = np.array([197, 148, 31])

        # Start with a transparent canvas
        frame_arr = np.zeros((image_diameter, image_diameter, 4), dtype=np.uint8)

        for y in range(image_diameter):
            for x in range(image_diameter):
                dx, dy = x - center, y - center
                distance_sq = dx**2 + dy**2

                # Draw the frame area
                if frame_inner_radius**2 < distance_sq <= self.frame_radius**2:
                    distance = math.sqrt(distance_sq)
                    dist_from_inner_edge = distance - frame_inner_radius
                    ridge_pos = (dist_from_inner_edge / self.frame_width) - 0.5
                    ridge_val = (math.cos(ridge_pos * math.pi) + 1) / 2
                    ridge_multiplier = 0.4 + ridge_val * 1.5

                    light_multiplier = 1.0
                    if self.light_angle is not None:
                        pixel_angle = math.atan2(-dy, dx)
                        angle_diff = math.cos(pixel_angle - self.light_angle)
                        light_multiplier = 1.0 + angle_diff * 0.7

                    final_multiplier = ridge_multiplier * light_multiplier
                    frame_color = np.clip(gold_color * final_multiplier, 0, 255).astype(np.uint8)
                    frame_arr[y, x] = [frame_color[0], frame_color[1], frame_color[2], 255]
        
        return Image.fromarray(frame_arr)

    def redraw(self, fast=False):
        image_diameter = self.image_diameter
        composite_img = Image.new("RGBA", (image_diameter, image_diameter), self.bg_color)

        scaled_width = int(self.original_img.width * self.zoom)
        scaled_height = int(self.original_img.height * self.zoom)
        
        # Optimization: Use NEAREST for fast updates (dragging), BICUBIC for quality
        resample_mode = Image.NEAREST if fast else Image.BICUBIC
        scaled_img = self.original_img.resize((scaled_width, scaled_height), resample_mode)
        
        paste_x = (image_diameter - scaled_width) // 2 + self.offset_x
        paste_y = (image_diameter - scaled_height) // 2 + self.offset_y
        self.scaled_img = scaled_img
        self.paste_x = paste_x
        self.paste_y = paste_y
        composite_img.paste(self.fill_layer, (0, 0), self.fill_layer)
        composite_img.paste(scaled_img, (paste_x, paste_y), scaled_img if scaled_img.mode == 'RGBA' else None)

        composite_img.paste(self.frame_overlay, (0, 0), self.frame_overlay) 
        self.final_image_to_save = composite_img

        # Optimization: NEAREST for display is usually sufficient and much faster
        display_image = self.final_image_to_save.resize((self.display_size, self.display_size), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        if self.selection_canvas_box:
            x0, y0, x1, y1 = self.selection_canvas_box
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="cyan", width=2, dash=(3, 2))

    def zoom_in(self, event, small_step=True):
        if small_step:
            self.zoom *= 1.05 # Slightly larger step for snappiness
        else:
            self.zoom *= 2.0 # Large zoom (100% increase)
        self.redraw()

    def zoom_out(self, event, small_step=True):
        if small_step:
            self.zoom /= 1.05 # Slightly larger step
        else:
            self.zoom /= 2.0 # Large zoom
        self.redraw()

    def pan_up(self, event): self.offset_y -= self.pan_step; self.redraw()
    def pan_down(self, event): self.offset_y += self.pan_step; self.redraw()
    def pan_left(self, event): self.offset_x -= self.pan_step; self.redraw()
    def pan_right(self, event): self.offset_x += self.pan_step; self.redraw()

    def _space_down(self, event): self.space_pan = True
    def _space_up(self, event): self.space_pan = False

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        scale = self.image_diameter / self.display_size
        return int(canvas_x * scale), int(canvas_y * scale)

    def start_left_action(self, event):
        self.left_dragged = False
        if self.space_pan:
            self.dragging = True
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
        else:
            self.selection_start_canvas = (event.x, event.y)
            self.selection_end_canvas = (event.x, event.y)
            self._update_selection_boxes()
            self.redraw()

    def do_left_drag(self, event):
        if self.space_pan:
            if not self.dragging:
                return
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            scale = self.image_diameter / self.display_size
            self.offset_x += int(dx * scale)
            self.offset_y += int(dy * scale)
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            self.redraw(fast=True)
        else:
            if self.selection_start_canvas is None:
                return
            self.selection_end_canvas = (event.x, event.y)
            if abs(event.x - self.selection_start_canvas[0]) > 2 or abs(event.y - self.selection_start_canvas[1]) > 2:
                self.left_dragged = True
            self._update_selection_boxes()
            self.redraw(fast=True)

    def stop_left_action(self, event):
        if self.space_pan:
            self.dragging = False
        else:
            if self.selection_start_canvas is None:
                return
            self.selection_end_canvas = (event.x, event.y)
            self._update_selection_boxes()
            if self.left_dragged and self.selection_image_box:
                self._save_selection_tile()
                self.status_label.config(text="Tile Selected! Click any void to fill.", fg="#005500")
                self.canvas.config(cursor="crosshair")
            elif self.sample_tile:
                img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
                if (img_x - self.frame_radius) ** 2 + (img_y - self.frame_radius) ** 2 <= self.frame_radius ** 2:
                    self._push_history()
                    self._bucket_fill(img_x, img_y, tile=self.sample_tile)
                    self.redraw(fast=False)
            self.redraw(fast=False)

    def _update_selection_boxes(self):
        if self.selection_start_canvas is None or self.selection_end_canvas is None:
            self.selection_canvas_box = None
            self.selection_image_box = None
            return

        x0, y0 = self.selection_start_canvas
        x1, y1 = self.selection_end_canvas
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        # Clamp to canvas bounds
        x0 = max(0, min(self.display_size, x0))
        x1 = max(0, min(self.display_size, x1))
        y0 = max(0, min(self.display_size, y0))
        y1 = max(0, min(self.display_size, y1))

        if x1 - x0 < 3 or y1 - y0 < 3:
            self.selection_canvas_box = None
            self.selection_image_box = None
            return

        self.selection_canvas_box = (x0, y0, x1, y1)
        self.selection_image_box = self._canvas_box_to_image_box(self.selection_canvas_box)

    def _canvas_box_to_image_box(self, box):
        x0, y0, x1, y1 = box
        ix0, iy0 = self._canvas_to_image_coords(x0, y0)
        ix1, iy1 = self._canvas_to_image_coords(x1, y1)
        ix0, ix1 = sorted((ix0, ix1))
        iy0, iy1 = sorted((iy0, iy1))
        ix0 = max(0, min(self.image_diameter, ix0))
        iy0 = max(0, min(self.image_diameter, iy0))
        ix1 = max(0, min(self.image_diameter, ix1))
        iy1 = max(0, min(self.image_diameter, iy1))
        return (ix0, iy0, ix1, iy1)

    def _push_history(self):
        if len(self.fill_history) >= 10:
            self.fill_history.pop(0)
        self.fill_history.append(self.fill_layer.copy())

    def undo_fill(self, event=None):
        if not self.fill_history:
            return
        self.fill_layer = self.fill_history.pop()
        self.redraw()

    def _save_selection_tile(self):
        if not self.selection_image_box:
            return
        x0, y0, x1, y1 = self.selection_image_box
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        tile = self.final_image_to_save.crop((x0, y0, x1, y1)).convert("RGBA")
        self.sample_tile = tile
        self.sample_tile_origin = (x0, y0)
        self._show_sample_preview(tile)

    def _show_sample_preview(self, tile):
        # Update the embedded canvas instead of a popup
        preview = tile.resize((40, 40), Image.NEAREST)
        self.sample_preview_tk = ImageTk.PhotoImage(preview)
        self.sample_canvas.delete("all")
        self.sample_canvas.create_image(0, 0, anchor=tk.NW, image=self.sample_preview_tk)

    def handle_fill_click(self, event):
        img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
        if (img_x - self.frame_radius) ** 2 + (img_y - self.frame_radius) ** 2 > self.frame_radius ** 2:
            return  # Outside the circular frame
        if self.sample_tile is None:
            messagebox.showinfo("Fill", "No sample tile yet. Left-drag to select a region first.")
            return
        self._push_history()
        self._bucket_fill(img_x, img_y, tile=self.sample_tile)
        self.redraw()

    def _bucket_fill(self, image_x, image_y, color=None, tile=None):
        # Create a composite image representing the current visual state (barriers)
        # This ensures we only fill the contiguous area bounded by the image and frame.
        
        if not hasattr(self, 'scaled_img'):
            return

        # 1. Construct the barrier map
        work_img = self.fill_layer.copy()
        
        # Paste the current scaled image (the subject) to act as a barrier
        if self.scaled_img:
            work_img.paste(self.scaled_img, (self.paste_x, self.paste_y), self.scaled_img)
            
        # Paste the frame overlay to constrain fill to the circle
        work_img.paste(self.frame_overlay, (0, 0), self.frame_overlay)

        # Snapshot before fill
        before_arr = np.array(work_img)

        # 2. Perform flood fill
        # Use a marker color that is likely unique (or just different from target)
        marker = (255, 0, 255, 255) # Magenta
        target_pixel = work_img.getpixel((image_x, image_y))
        if target_pixel == marker:
            marker = (0, 255, 0, 255) # Green

        ImageDraw.floodfill(work_img, (image_x, image_y), marker, thresh=0)

        # 3. Calculate mask (pixels that changed)
        after_arr = np.array(work_img)
        mask = np.any(before_arr != after_arr, axis=-1)

        if mask is None or not np.any(mask): return

        new_arr = np.array(self.fill_layer)
        if tile:
            tile_arr = np.array(tile)
            th, tw = tile_arr.shape[:2]
            ox, oy = self.sample_tile_origin
            rows, cols = np.where(mask)
            tile_y = (rows - oy) % th
            tile_x = (cols - ox) % tw
            new_arr[rows, cols] = tile_arr[tile_y, tile_x]
        elif color:
            fill_rgba = (*color[:3], 255) if len(color) == 4 else (*color, 255)
            new_arr[mask] = fill_rgba

        self.fill_layer = Image.fromarray(new_arr, mode="RGBA")

    def save_and_close(self):
        # Before saving, create a final circular mask to ensure everything
        # outside the frame is transparent.
        image_diameter = self.frame_radius * 2
        final_mask = Image.new("L", (image_diameter, image_diameter), 0)
        mask_arr = np.array(final_mask)
        center = self.frame_radius
        radius_sq = self.frame_radius**2

        for y in range(image_diameter):
            for x in range(image_diameter):
                if (x - center)**2 + (y - center)**2 <= radius_sq:
                    mask_arr[y, x] = 255
        
        final_mask = Image.fromarray(mask_arr)
        self.final_image_to_save.putalpha(final_mask)

        self.final_image_to_save.save(self.output_path)
        print(f"Saved adjusted image to '{self.output_path}'")
        self.master.destroy()

def launch_interactive_editor(image, output_path, frame_radius, frame_width, bg_color, light_angle):
    root = tk.Tk()
    root.title("Interactive Image Adjuster")
    app = InteractiveImageEditor(root, image, output_path, frame_radius, frame_width, bg_color, light_angle)
    root.mainloop()

def make_images_circular(input_dir="images", output_dir="images_circular", frame_width_percent=5.0, light_source="NW", overlay_frame=False):
    """
    Finds all PNG images in an input directory, crops them to their
    content, and transforms them into a circular shape, saving them
    to an output directory.

    Args:
        input_dir (str): Directory containing the source PNG images.
        output_dir (str): Directory where circular images will be saved.
        frame_width_percent (float): Width of the golden frame as a percentage
                                     of the image's diagonal. If 0, no frame.
        light_source (str): Direction of the light source for the frame's
                            3D effect. Accepts "NW", "NE", "SW", "SE", "CE"
                            (Center/Top-down). (Note: overlay_frame is deprecated
                            in favor of a more robust interactive mode).
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        print("Please create an 'images' directory and place your PNG files inside it.")
        return

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    if not image_files:
        print(f"No PNG images found in the '{input_dir}' directory.")
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    # Define light source angles in radians for lighting calculation
    light_angles = {
        "NW": math.radians(135),
        "NE": math.radians(45),
        "SW": math.radians(225),
        "SE": math.radians(315),
        "CE": None  # Center/Top-down light has no angle
    }
    light_angle = light_angles.get(light_source.upper())

    print(f"Found {len(image_files)} PNG image(s) to process.")
    for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path).convert("RGBA") as img:
                    # Find the tightest bounding box of non-transparent pixels.
                    bbox = img.getbbox()
                    if not bbox:
                        print(f"Skipping '{filename}' as it is fully transparent.")
                        continue

                    # Crop the image to the content.
                    cropped_img = img.crop(bbox)

                    # Interactive squaring logic
                    if cropped_img.width != cropped_img.height:
                        print(f"Image '{filename}' is not square ({cropped_img.width}x{cropped_img.height}).")
                        print("The circular frame will be drawn first; fill empty areas inside the editor with a right-click.")

                    width, height = cropped_img.size

                    # a) Evaluate if the picture has significant features.
                    # We check if the cropped image has more than a handful of colors.
                    # A low number might indicate a simple icon or a blank image.
                    colors = cropped_img.getcolors(width * height)
                    if colors and len(colors) <= 4:
                        print(f"Skipping '{filename}' as it lacks significant features (<= 4 colors).")
                        continue
 
                    # Calculate the radius based on content shape. For circular images, this
                    # will be tight; for rectangular, it will be half the diagonal.
                    pic_radius = get_content_radius(cropped_img)
                    pic_diameter = pic_radius * 2
                    
                    # Calculate frame width in pixels from the percentage
                    frame_width = math.ceil(pic_diameter * (frame_width_percent / 100.0))
 
                    # The frame is now always added around the picture content.
                    total_diameter = pic_diameter + (frame_width * 2)
                    frame_inner_radius = pic_radius
 
                    total_radius = total_diameter / 2
 
                    # b) Determine a fill color for the background.
                    # First, try to get a background color from the edges.
                    bg_color = get_background_color(cropped_img, default_color=None)

                    # If no edge color was found, calculate the average color of the content itself.
                    # This is used to fill transparent gaps between the content and the frame.
                    if bg_color is None:
                        opaque_pixels = [p for p in cropped_img.getdata() if p[3] > 128]
                        if opaque_pixels:
                            avg_r = sum(p[0] for p in opaque_pixels) // len(opaque_pixels)
                            avg_g = sum(p[1] for p in opaque_pixels) // len(opaque_pixels)
                            avg_b = sum(p[2] for p in opaque_pixels) // len(opaque_pixels)
                            bg_color = (avg_r, avg_g, avg_b, 255)
                        else: # Fallback for fully transparent cropped images
                            bg_color = (51, 51, 51, 255)

                    # Create the final canvas for the automatic result.
                    # The area between the image and the frame is filled with the bg_color.
                    final_img = Image.new("RGBA", (total_diameter, total_diameter), bg_color)
                    # Paste the cropped image in the center
                    final_img.paste(cropped_img, ((total_diameter - width) // 2, (total_diameter - height) // 2), cropped_img)

                    # Create a circular mask and apply it
                    mask = Image.new('L', (total_diameter, total_diameter), 0)
                    mask_arr = np.array(mask)
                    center = total_radius
                    radius_sq = total_radius**2
                    for y in range(total_diameter):
                        for x in range(total_diameter):
                            if (x - center)**2 + (y - center)**2 <= radius_sq:
                                mask_arr[y, x] = 255
                    mask = Image.fromarray(mask_arr)
                    final_img.putalpha(mask)

                    # Ask user for action
                    while True:
                        prompt = (f"\nAction for '{filename}':\n"
                                  "  (y) Save automatically\n"
                                  "  (v) View automatic result\n"
                                  "  (e) Edit manually\n" 
                                  "  (s) Skip this file [default]\n"
                                  "  (q) Quit program\n"
                                  "Choose an option (y/v/e/q) or press Enter to skip: ")
                        answer = input(prompt).lower()

                        if answer in ['y', 'yes']:
                            final_img.save(output_path)
                            print(f"Saved '{filename}' to '{output_path}'")
                            break
                        elif answer in ['v', 'view']:
                            print("Showing automatic result. Close the image window to continue...")
                            final_img.show(title=f"Automatic Result for {filename}")
                            # Loop continues, asking for a new action after viewing.
                        elif answer in ['e', 'edit']: # pragma: no cover
                            print("Launching interactive editor... (Left-drag to select a tile, left-click to fill, hold Space+drag or arrows to pan, right-click also fills, 'z'/'o' to zoom, 'u' to undo fills)")
                            # The interactive editor places the frame around the content,
                            # so the frame radius is the picture radius.
                            launch_interactive_editor(cropped_img, output_path, pic_radius, frame_width, bg_color, light_angle)
                            break
                        elif answer in ['s', 'skip', '']: # Empty string for Enter key
                            print(f"Skipped '{filename}'.")
                            break
                        elif answer in ['q', 'quit']:
                            print("Quitting program.")
                            return # Exit the entire function
                        else:
                            print("Invalid input. Please choose one of the options.")

            except Exception as e:
                print(f"Could not process '{filename}'. Error: {e}")

if __name__ == "__main__":
    # Check for tkinter availability before starting
    try:
        import tkinter
    except ImportError:
        print("Error: tkinter is not installed or available.")
        print("Please install it to run the interactive editor (e.g., 'pip install tk').")
        exit()
    # Assuming your images are in a subfolder named 'images'
    # relative to where you run this script.

    make_images_circular(frame_width_percent=5.0, light_source="NW")
