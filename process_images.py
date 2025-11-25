import os
from collections import Counter
from PIL import Image, ImageTk
import numpy as np
import math
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk
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

        # Determine a sensible canvas size that fits the screen
        max_dim = self.frame_radius * 2
        # Limit to 1/4 of screen real estate by using 50% of height
        screen_height = master.winfo_screenheight() 
        self.display_size = min(max_dim, int(screen_height * 0.5))
        
        self.canvas_size = self.display_size
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Add a frame for controls
        control_frame = tk.Frame(master)
        control_frame.pack(pady=5)

        save_button = tk.Button(control_frame, text="Save and Close", command=self.save_and_close)
        save_button.pack(side=tk.LEFT, padx=5)

        # Bind keys for interaction
        # Use 'z' for zoom in to avoid conflict with 'i' in input fields
        master.bind("<KeyPress-z>", self.zoom_in)
        master.bind("<KeyPress-o>", self.zoom_out)
        master.bind("<KeyPress-Up>", self.pan_up)
        master.bind("<KeyPress-Down>", self.pan_down)
        master.bind("<KeyPress-Left>", self.pan_left)
        master.bind("<KeyPress-Right>", self.pan_right)

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

    def redraw(self):
        image_diameter = self.frame_radius * 2
        
        # 1. Create a transparent background layer. The bg_color will fill the space
        #    between the image and the frame.
        composite_img = Image.new("RGBA", (image_diameter, image_diameter), self.bg_color)

        # 2. Paste the scaled and panned user image onto the composite image
        scaled_width = int(self.original_img.width * self.zoom)
        scaled_height = int(self.original_img.height * self.zoom)
        scaled_img = self.original_img.resize((scaled_width, scaled_height), Image.LANCZOS)
        paste_x = (image_diameter - scaled_width) // 2 + self.offset_x
        paste_y = (image_diameter - scaled_height) // 2 + self.offset_y
        composite_img.paste(scaled_img, (paste_x, paste_y), scaled_img if scaled_img.mode == 'RGBA' else None)

        # 3. Paste the pre-rendered frame on top
        # The frame overlay already has transparency, so it composites correctly.
        composite_img.paste(self.frame_overlay, (0, 0), self.frame_overlay) 
        self.final_image_to_save = composite_img

        # 4. Scale the result for display in the GUI and update the canvas
        display_image = self.final_image_to_save.resize((self.display_size, self.display_size), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def zoom_in(self, event):
        self.zoom *= 1.1
        self.redraw()

    def zoom_out(self, event):
        self.zoom /= 1.1
        self.redraw()

    def pan_up(self, event): self.offset_y -= self.pan_step; self.redraw()
    def pan_down(self, event): self.offset_y += self.pan_step; self.redraw()
    def pan_left(self, event): self.offset_x -= self.pan_step; self.redraw()
    def pan_right(self, event): self.offset_x += self.pan_step; self.redraw()

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
                        elif answer in ['e', 'edit']:
                            print("Launching interactive editor... (Use arrows to pan, 'z'/'o' to zoom)")
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
