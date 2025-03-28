from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

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
        print(font_path,end=" font path\n")
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
        font = ImageFont.truetype(font_path, width // 40)  # Adjust font size if needed; default == width // 40

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

def annotate_image(image_or_image_path, save_path = None, font_path = 'fonts/arial.tff',
                   box_width = None, box_height = None, box_coords = None, mask = None):
    with open_image(image_or_image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width, height = img.size
        # 将 PIL Image 转换为 numpy 数组
        img = np.array(img)

    # uniformly sample points from the mask
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
    font = ImageFont.truetype(font_path, width // 40)  # Adjust font size if needed; default == width // 40
    mask_array = np.array(mask)
    mask_coords = np.argwhere(mask_array == 1)
    mask_coords = mask_coords.astype(np.int32)
    # sample 10 points from the mask
    sampled_coords = mask_coords[np.random.choice(mask_coords.shape[0], 10, replace=False)]
    # draw the points on the image transparently a transparent circle and a number inside the circle
    # eliminate the points that are too close to each other
    deleted = np.zeros(len(sampled_coords), dtype=np.bool_)
    for i, coord in enumerate(sampled_coords):
        if not deleted[i]:
            for j, other_coord in enumerate(sampled_coords):
                if i != j and not deleted[j]:
                    if np.linalg.norm(coord - other_coord) < 25:
                        deleted[i] = True
                        break
    print(deleted,end="  deleted\n")
    sampled_coords = sampled_coords[~deleted]


    print(sampled_coords,end="  sampled_coords\n")
    for i, coord in enumerate(sampled_coords):
        print(coord,end="  coord\n")
        overlay = img.copy()
        # 让圆有黑边，内部是白色
        overlay = cv2.circle(overlay, (coord[1], coord[0]), 18, (0, 0, 0), 2)  # 外圈黑色，细线
        # overlay = cv2.circle(overlay, (coord[1], coord[0]), 3, (255, 255, 255), -1)  # 中心白色
        font_size = 0.8
        alpha = 0.5  # 增加不透明度
        img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        overlay = img.copy()
        # 计算文字大小以居中显示
        text = str(i)
        font_scale = font_size
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = coord[1] - text_width // 2
        text_y = coord[0] + text_height // 2
        overlay = cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        alpha = 0.7  # 增加文字不透明度
        img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    if save_path is not None:
        print(">>> annotated image processed, stored in", save_path)
        cv2.imwrite(save_path, img)
    return img, sampled_coords

