from PIL import Image, ImageDraw, ImageFont
import numpy as np

def open_image(image_or_image_path):
    if isinstance(image_or_image_path, Image.Image):
        return image_or_image_path
    elif isinstance(image_or_image_path, str):
        return Image.open(image_or_image_path)
    else:
        raise ValueError("Unsupported input type!")

def dot_matrix_two_dimensional(image_or_image_path, save_path = None, dots_size_w = 6, dots_size_h = 6, save_img = False, font_path = 'fonts/arial.tff', 
                               box_width = None, box_height = None, box_coords = None):
    """
    takes an original image as input, save the processed image to save_path. Each dot is labeled with two-dimensional Cartesian coordinates (x,y). Suitable for single-image tasks.
    control args:
    1. dots_size_w: the number of columns of the dots matrix
    2. dots_size_h: the number of rows of the dots matrix
    """
    with open_image(image_or_image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img, 'RGB')

        width, height = img.size
        grid_size_w = dots_size_w + 1
        grid_size_h = dots_size_h + 1
        cell_width = width / grid_size_w
        cell_height = height / grid_size_h

        font = ImageFont.truetype(font_path, width // 60)  # Adjust font size if needed; default == width // 40

        count = 0
        for j in range(1, grid_size_h):
            for i in range(1, grid_size_w):
                x = int(i * cell_width)
                y = int(j * cell_height)

                pixel_color = img.getpixel((x, y))
                # choose a more contrasting color from black and white
                if pixel_color[0] + pixel_color[1] + pixel_color[2] >= 255 * 3 / 2:
                    opposite_color = (0,0,0)
                else:
                    opposite_color = (255,255,255)

                circle_radius = width // 240  # Adjust dot size if needed; default == width // 240
                draw.ellipse([(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)], fill=opposite_color)

                text_x, text_y = x + 3, y
                count_w = count // dots_size_w
                count_h = count % dots_size_w
                label_str = f"({count_w+1},{count_h+1})"
                draw.text((text_x, text_y), label_str, fill=opposite_color, font=font)
                count += 1
        if save_img:
            print(">>> dots overlaid image processed, stored in", save_path)
            img.save(save_path)
        return img
    
def dot_matrix_two_dimensional_with_box(image_or_image_path, save_path = None, dots_size_w = 6, dots_size_h = 6, save_img = False, font_path = 'fonts/arial.tff', 
                            box_width = None, box_height = None, box_coords = None):
    MAX_GLOBAL_GRID = 25
    with open_image(image_or_image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img, 'RGB')

        width, height = img.size
        grid_size_w = dots_size_w - 1
        grid_size_h = dots_size_h - 1
        cell_width = box_width / grid_size_w
        cell_height = box_height / grid_size_h
        
        if cell_width <= width / MAX_GLOBAL_GRID:
            cell_width = width / MAX_GLOBAL_GRID
            grid_size_w = int(np.floor(box_width / cell_width))
            dots_size_w = grid_size_w + 1
            cell_width = box_width / grid_size_w
        if cell_height <= height / MAX_GLOBAL_GRID:
            cell_height = height / MAX_GLOBAL_GRID
            grid_size_h = int(np.floor(box_height / cell_height))
            dots_size_h = grid_size_h + 1
            cell_height = box_height / grid_size_h

        font = ImageFont.truetype(font_path, width // (MAX_GLOBAL_GRID * 2))  # Adjust font size if needed; default == width // 40

        if box_coords is None:
            box_coords = (width // 2, height // 2)
        if box_width is None:
            box_width = width // 2
        if box_height is None:
            box_height = height // 2

        box_x, box_y = box_coords
        box_left = box_x - box_width // 2
        box_right = box_x + box_width // 2
        box_top = box_y - box_height // 2
        box_bottom = box_y + box_height // 2

        count = 0
        # eps = 1e-3  # to avoid out of range
        for j in range(0, dots_size_h):
            for i in range(0, dots_size_w):
                x = int(i * cell_width) + box_left
                y = int(j * cell_height) + box_top

                # if box_left <= x <= box_right + eps and box_top <= y <= box_bottom + eps:
                pixel_color = img.getpixel((x, y))
                # choose a more contrasting color from black and white
                if pixel_color[0] + pixel_color[1] + pixel_color[2] >= 255 * 3 / 2:
                    opposite_color = (0,0,0)
                else:
                    opposite_color = (255,255,255)

                circle_radius = width // 240  # Adjust dot size if needed; default == width // 240
                draw.ellipse([(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)], fill=opposite_color)

                text_x, text_y = x + 3, y
                count_w = count // dots_size_w
                count_h = count % dots_size_w
                label_str = f"({count_w+1},{count_h+1})"
                draw.text((text_x, text_y), label_str, fill=opposite_color, font=font)
                count += 1

        if save_img:
            print(">>> dots overlaid image processed, stored in", save_path)
            img.save(save_path)
        return img, (box_left, box_top), (cell_width, cell_height)