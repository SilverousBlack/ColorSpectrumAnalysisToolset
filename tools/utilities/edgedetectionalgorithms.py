from PIL import Image, ImageFilter

def ImprovedSecondDerivativeEdgeDetection(target: Image.Image):
    width, height = target.size
    blur_radius = int((min(width, height) * 0.05) if (min(width, height) * 0.05) > 0 else (min(width, height) * 0.1))
    copy = target.filter(ImageFilter.UnsharpMask(blur_radius)).filter(ImageFilter.GaussianBlur(blur_radius))
    copy = (ImageEnhance.Contrast(target).enhance(1.75)).filter(ImageFilter.FIND_EDGES).convert("LA")
    copy = np.array(copy, np.uint32)[:, :, 0]
    edgepx = copy[:, :].max() + 1
    # edge denoising routine
    #for i in range(height):
    #    for j in range(width):
    #        if copy[i, j] >= (edgepx * 0.5):
    #            copy[i, j] = 255
    #        elif (edgepx * 0.25) <= copy[i, j] < (edgepx * 0.5):
    #            copy[i, j] = 127
    #        elif (edgepx * 0.125) <= copy[i, j] < (edgepx * 0.25):
    #            copy[i, j] = 63
    #        else:
    #           copy[i, j] = 0
    copy[copy >= (edgepx * 0.5)] = 255
    copy[(copy >= (edgepx * 0.25)) & (copy < edgepx * 0.5)] = 127
    copy[(copy >= (edgepx * 0.125)) & (copy < (edgepx * 0.25))] = 63
    copy[copy < (edgepx * 0.125)] = 0
    # probability edge enhancement with edge denoising
    internal = np.zeros((height, width), np.uint32)
    for i in range(height):
        for j in range(width):
            density_prob = 0.0
            try:
                density_prob += ((copy[i - 1, j - 1] + 1) // 64) / 4
                density_prob += ((copy[i, j - 1] + 1) // 64) / 4
                density_prob += ((copy[i + 1, j - 1] + 1) // 64) / 4
                density_prob += ((copy[i - 1, j] + 1) // 64) / 4
                density_prob += ((copy[i, j] + 1) // 64) / 4
                density_prob += ((copy[i + 1, j] + 1) // 64) / 4
                density_prob += ((copy[i - 1, j + 1] + 1) // 64) / 4
                density_prob += ((copy[i, j + 1] + 1) // 64) / 4
                density_prob += ((copy[i + 1, j + 1] + 1) // 64) / 4
            except:
                pass
            if density_prob >= 3:
                internal[i, j] = 255
            elif 2 <= density_prob < 3:
                internal[i, j] = 127
            elif 1 <= density_prob < 2:
                internal[i, j] = 63
            else:
                internal[i, j] = 0
    internal = np.expand_dims(internal, axis=2)
    internal = np.insert(internal, 1, 255, axis=2).astype('uint8')
    internal = Image.fromarray(internal, 'LA')
    return internal.copy()