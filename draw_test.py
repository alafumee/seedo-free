from PIL import Image, ImageDraw, ImageFont

# 创建一个带有透明背景的图像
width, height = 200, 200
image = Image.new("RGBA", (width, height), (255, 255, 255, 0))

draw = ImageDraw.Draw(image)

# 设置圆的坐标
x0, y0, x1, y1 = 50, 50, 150, 150

# 画一个半透明的圆
circle_color = (255, 0, 0, 128)  # 红色，透明度128（0-255）
draw.ellipse([x0, y0, x1, y1], fill=circle_color)

# 设置字体（使用默认字体）
font_size = 40
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

# 计算文字位置
text = "5"
text_width, text_height = draw.textsize(text, font=font)
text_x = (x0 + x1) // 2 - text_width // 2
text_y = (y0 + y1) // 2 - text_height // 2

# 画文字（黑色，不透明）
draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255))

# 显示图片
image.show()
