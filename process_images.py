import os
from PIL import Image
import numpy as np
import math

def get_background_color(image, default_color=(0, 0, 0, 0)):
    """
    Determines the predominant background color of an image by checking its corners.
    It assumes the subject is centered and the corners represent the background.

    Args:
        image (PIL.Image.Image): The image to analyze.
        default_color (tuple): The color to return if analysis fails.

    Returns:
        tuple: An RGBA tuple representing the most common color in the corners.
    """
    try:
        width, height = image.size
        # Sample a 1x1 pixel from each corner
        corners = [image.getpixel((0, 0)), image.getpixel((width - 1, 0)),
                   image.getpixel((0, height - 1)), image.getpixel((width - 1, height - 1))]
        # Find the most frequent color among the corners
        return max(set(corners), key=corners.count)
    except (IndexError, ValueError):
        return default_color

def make_images_circular(input_dir="images", output_dir="images_circular", frame_width=20, light_source="NW", overlay_frame=True):
    """
    Finds all PNG images in an input directory, crops them to their
    content, and transforms them into a circular shape, saving them
    to an output directory.

    Args:
        input_dir (str): Directory containing the source PNG images.
        output_dir (str): Directory where circular images will be saved.
        frame_width (int): Width of the golden frame in pixels. If 0, no frame.
        light_source (str): Direction of the light source for the frame's
                            3D effect. Accepts "NW", "NE", "SW", "SE", "CE"
                            (Center/Top-down).
        overlay_frame (bool): If True, the frame is drawn on top of the image's
                              outer edge. If False, it's added around the image.
    """
    if not os.path.isdir(input_dir):
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

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
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
 
                    # To ensure the entire image content fits within the circle without
                    # clipping corners, the diameter must be the diagonal of the content.
                    # We'll create a square canvas of this diagonal size.
                    pic_diameter = math.ceil(math.sqrt(width**2 + height**2))
                    pic_radius = pic_diameter / 2
 
                    # Create a new square image with a transparent background
                    square_img = Image.new("RGBA", (pic_diameter, pic_diameter), (0, 0, 0, 0))
                    # Calculate coordinates to paste the cropped image at the center
                    paste_x = (pic_diameter - width) // 2
                    paste_y = (pic_diameter - height) // 2
                    square_img.paste(cropped_img, (paste_x, paste_y))
 
                    # Determine total size and frame boundaries based on overlay setting
                    if overlay_frame:
                        total_diameter = pic_diameter
                        frame_inner_radius = pic_radius - frame_width
                    else:
                        total_diameter = pic_diameter + (frame_width * 2)
                        frame_inner_radius = pic_radius
 
                    total_radius = total_diameter / 2
 
                    # b) Determine the predominant background color from the cropped image.
                    bg_color = get_background_color(cropped_img)

                    # Create the final canvas. Initialize with transparency.
                    final_img = Image.new("RGBA", (total_diameter, total_diameter), (0, 0, 0, 0))
                    
                    source_arr = np.array(square_img)
                    final_arr = np.array(final_img)

                    # Center of the new circular image
                    center_x, center_y = total_radius, total_radius

                    # Base color for the golden frame
                    gold_color = np.array([197, 148, 31]) # A richer, deeper gold color

                    for y in range(total_diameter):
                        for x in range(total_diameter):
                            # Calculate distance from the center of the new image
                            dx, dy = x - center_x, y - center_y
                            distance_sq = dx**2 + dy**2 # Use squared distance for performance

                            # Pre-calculate squared radii for comparison
                            pic_radius_sq = pic_radius**2
                            frame_inner_radius_sq = frame_inner_radius**2 if frame_inner_radius > 0 else 0
                            total_radius_sq = total_radius**2

                            # Fill background color inside the frame area first
                            if distance_sq <= total_radius_sq:
                                final_arr[y, x] = bg_color

                            # --- Draw the circular picture content ---
                            # Check if the pixel is within the picture's circular area
                            if distance_sq <= pic_radius_sq:
                                # Map the point in the circle back to the square source image.
                                # Since both are circles/squares centered at radius, the mapping is direct.
                                orig_x = int(pic_radius + dx)
                                orig_y = int(pic_radius + dy)
                                
                                # Draw the picture pixel, but only if it's not transparent
                                # and it's inside the frame's inner boundary (or if frame overlays)
                                if not (overlay_frame and distance_sq > frame_inner_radius_sq):
                                    pixel = source_arr[orig_y, orig_x]
                                    if pixel[3] > 0: # Only draw non-transparent pixels
                                        final_arr[y, x] = pixel

                            # --- Draw the golden frame ---
                            # The frame is drawn between its inner radius and the total radius
                            if frame_width > 0 and frame_inner_radius_sq < distance_sq <= total_radius_sq:
                                distance = math.sqrt(distance_sq) # Calculate true distance only when needed
                                # 1. Calculate ridge effect (brighter in the middle of the frame)
                                dist_from_inner_edge = distance - frame_inner_radius
                                # Normalize from -0.5 (inner) to 0.5 (outer) across the frame width
                                ridge_pos = (dist_from_inner_edge / frame_width) - 0.5
                                # Use cosine for a smooth curve, peaking at the center (ridge_pos=0)
                                ridge_val = (math.cos(ridge_pos * math.pi) + 1) / 2  # Range 0..1
                                ridge_multiplier = 0.4 + ridge_val * 1.5 # Remap to 0.4..1.9 for very high contrast

                                # 2. Calculate lighting effect (highlight/shadow)
                                light_multiplier = 1.0
                                if light_angle is not None: # Not 'CE'
                                    pixel_angle = math.atan2(-dy, dx)
                                    # Cosine of angle difference determines highlight
                                    angle_diff = math.cos(pixel_angle - light_angle) # Range -1..1
                                    # Map cos result to a stronger brightness multiplier (e.g., 0.3..1.7)
                                    light_multiplier = 1.0 + angle_diff * 0.7

                                # 3. Combine effects and calculate final color
                                final_multiplier = ridge_multiplier * light_multiplier
                                frame_color = np.clip(gold_color * final_multiplier, 0, 255).astype(np.uint8)
                                
                                # Assign RGBA value
                                final_arr[y, x] = [frame_color[0], frame_color[1], frame_color[2], 255]

                    # Convert the numpy array back to an image and save it
                    result_img = Image.fromarray(final_arr)
                    result_img.save(output_path)
                    print(f"Processed '{filename}' and saved to '{output_path}'")

            except Exception as e:
                print(f"Could not process '{filename}'. Error: {e}")

if __name__ == "__main__":
    # Assuming your images are in a subfolder named 'images'
    # relative to where you run this script.
    make_images_circular(frame_width=20, light_source="NW", overlay_frame=True)
