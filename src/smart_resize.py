import math

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by "factor".
    2. The total number of pixels is within the range ["min_pixels", "max_pixels"].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


if __name__ == "__main__":
    
    origin_height = 2400
    origin_width = 1080
    
    MAX_PIXELS = 3600*28*28
    
    x = 123
    y = 456
    
    resized_h, resized_w = smart_resize(height = origin_height, width = origin_width, max_pixels=MAX_PIXELS)
    
    print(resized_h, resized_w)
    
    resized_x = x / origin_width * resized_w
    resized_y = y / origin_height * resized_h
    
    print(int(resized_x), int(resized_y))
