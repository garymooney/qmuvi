# Methods relating to the generation of video from sampled density matrices
import os
import math


import qmuvi.musical_processing as musical_processing


def filter_color_blend(pic, colour, alpha):

    """ assumes pic is 2d array of uint8. Returns weighted average of image pizel colours and given colour.

    """

    for i in range(len(pic)):

        for j in range(len(pic[i])):

            pic[i][j] = blend_colour(pic[i][j], colour, alpha)
    
    return pic


def blend_colour(colour1, colour2, alpha):

    return [((1 - alpha) * colour1[i] + (alpha) * colour2[i]) for i in range(min(len(colour2), len(colour1)))]


def filter_color_multiply(pic, colour):

    """ assumes pic is 2d array of uint8. Returns weighted average of image pizel colours and given colour.

    """

    for i in range(len(pic)):

        for j in range(len(pic[i])):

            pic[i][j] = [int(pic[i][j][k] * (colour[k]/255.0)) for k in range(min(len(colour), len(pic[i][j])))]
    
    return pic


def filter_colour_round_with_threshold(pic, threshold = 0.5, colour_dark = [0, 0, 0], colour_light = [255, 255, 255]):

        """ assumes pic is 2d array of uint8. Rounds each colour to black or white based on average RGB and threshold.

        """

        colour_array_dims = 0

        try:

            colour_array_dims = len(pic[0][0])

        except:

            colour_array_dims = 0


        if (colour_array_dims > 0): # pic is an RGB image (0-255)

            for i in range(len(pic)):

                for j in range(len(pic[i])):

                    average_rgb = np.mean([pic[i][j][k] for k in range(min(3, colour_array_dims))])

                    if average_rgb > threshold:

                        pic[i][j] = colour_light

                    else:

                        pic[i][j] = colour_dark

        else: # pic is a mask (0-1)

            for i in range(len(pic)):

                for j in range(len(pic[i])):

                    if pic[i][j] > threshold:

                        pic[i][j] = colour_light

                    else:

                        pic[i][j] = colour_dark
        return pic


def lerp(lerp_from, lerp_to, t):

    t = max(min(t, 1.0), 0.0)

    return [((1-t) * a) + (t * b) for a, b in zip(lerp_from, lerp_to)]


def ease_in(ease_from, ease_to, t):

    ''' slow at the beginning, fast at the end.'''

    t = max(min(t, 1.0), 0.0)

    scaled_t = 1 - math.cos(t * math.pi / 2.0)

    return [((1-scaled_t) * a) + (scaled_t * b) for a, b in zip(ease_from, ease_to)]


def ease_out(ease_from, ease_to, t):

    ''' fast at the beginning, slow at the end.'''

    t = max(min(t, 1.0), 0.0)

    scaled_t = math.sin(t * math.pi / 2.0)

    return [((1-scaled_t) * a) + (scaled_t * b) for a, b in zip(ease_from, ease_to)]


def invert_cmap(cmap):

    from matplotlib.colors import ListedColormap


    newcolors = cmap(np.linspace(0, 1, 256))

    for i in range(256):

        newcolors[i, :] = np.array([1-cmap(i/256)[0],1-cmap(i/256)[1],1-cmap(i/256)[2],1])

    return ListedColormap(newcolors)