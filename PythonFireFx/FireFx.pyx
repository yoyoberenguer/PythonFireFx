# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True,
# cython: optimize.use_switch=True
# encoding: utf-8

"""
                 GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

Copyright Yoann Berenguer
"""


__VERSION__ = "1.0.2"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

try:
    import numpy
    from numpy import empty, uint8, int16, float32, asarray, linspace, \
        ascontiguousarray, zeros, uint16, uint32, int32, int8
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

try:
    cimport cython
    from cython.parallel cimport prange

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy, HWACCEL, HWSURFACE, \
    QUIT, K_SPACE, BLEND_RGB_ADD, Rect, BLEND_RGB_MAX, BLEND_RGB_MIN
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, \
        make_surface, blit_array, pixels_red, \
    pixels_green, pixels_blue
    from pygame.image import frombuffer, fromstring, tostring
    from pygame.math import Vector2
    from pygame import _freetype
    from pygame._freetype import STYLE_STRONG, STYLE_NORMAL
    from pygame.transform import scale, smoothscale, rotate, scale2x
    from pygame.pixelcopy import array_to_surface

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

try:
    cimport cython
    from cython.parallel cimport prange
    from cpython cimport PyObject_CallFunctionObjArgs, PyObject, \
        PyList_SetSlice, PyObject_HasAttr, PyObject_IsInstance, \
        PyObject_CallMethod, PyObject_CallObject
    from cpython.dict cimport PyDict_DelItem, PyDict_Clear, PyDict_GetItem, PyDict_SetItem, \
        PyDict_Values, PyDict_Keys, PyDict_Items
    from cpython.list cimport PyList_Append, PyList_GetItem, PyList_Size, PyList_SetItem
    from cpython.object cimport PyObject_SetAttr

except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

from libc.stdlib cimport rand

cdef int THREADS = 2

cdef float ONE_360 = <float> 1.0 / 360.0
cdef float ONE_255 = <float> 1.0 / 255.0




@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline shader_fire_effect(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] fire_,

        # OPTIONAL
        unsigned short int reduce_factor_ = 3,
        unsigned short int fire_intensity_= 32,
        bint smooth_                      = True,
        bint bloom_                       = True,
        bint fast_bloom_                  = True,
        unsigned char bpf_threshold_      = 0,
        unsigned int low_                 = 0,
        unsigned int high_                = 600,
        bint brightness_                  = True,
        float brightness_intensity_       = 0.15,
        object surface_                   = None,
        bint adjust_palette_              = False,
        tuple hsl_                        = (10, 80, 1.8),
        bint transpose_                   = False,
        bint border_                      = False,
        bint blur_                        = True
        ):
    """
    FIRE SHADER EFFECT 

    * FIRE TEXTURE SIZES 
    
    input width_  : integer,  
    input height_ : integer
    
    width_ and height_ values define the size of the texture e.g Surface(width x height)

    * FIRE ASPECT (CONTROL OVER THE WIDTH): 
    
    inputs low_ : integer  
    input high_ : integer 
    
    Optional arguments low_ & high_ (integer values) define the width 's limits of the fire effect. 
    low_ for the starting point and high_ for the ending of the effect.
    e.g low_ = 10 and high_ = 200. The fire effect will be contain within width = 10 and 200
    low_ & high_ values must be in range [0 ... width_]  
        
    * FIRE HEIGHT:
    
    input factor_ : float
    
    The fire maximum height can be adjust with the variable factor_ (float value)
    value > 3.95 will contain the effect within the display 
    value < 3.95 will enlarge the effect over the display height  
    Recommended value is 3.95 with reduce_factor_ = 3 otherwise adjust the value manually 
    to contain the fire effect within the display
        
    * SPEED CONSIDERATION
    
    input reduce_factor_ : integer
    
    The argument reduce_factor_ control the size of the texture to be processed 
    e.g : a value of 2, divide by 4 the pygame surface define by the values (width_ & height_)
    Smaller texture improve the overall performances but will slightly degrade the fire aspect, 
    especially if the blur and smooth option are not enabled.
    Recommended value for reduce_factor_ is 3 (fast process)   
    reduce_factor_ values must be an integer in range [ 0 ... 4] 
    The reduce_factor_ value will have a significant impact on the fire effect maximum height, 
    adjust the argument factor_ accordingly

    * FIRE INTENSITY AT THE SOURCE
    
    input fire_intensity_: integer
    
    Set the fire intensity with the variable fire_intensity_, 0 low flame,
    32 maximum flame effect
    Values must be an int in range [0 ... 32] 

    * SMOOTHING THE EFFECT
    
    input smooth_: True | False
    
    When smooth_ is True the algorithm will use the pygame function smoothscale (bi-linear 
    filtering) or False the final texture will be adjust with the scale function.
    Set this variable to False if you need the best performance for the effect or if you require
    a pixelated fire effect. Otherwise set the variable to True for a more realistic effect. 

    
    * BLOOM EFFECT 
    
    input bloom_         : True | False
    input fast_bloom_    : True | False
    input bpf_threshold_ : integer
       
    Fire effect produce a bright and smooth light effect to the background texture where the fire 
    intensity is at its maximum.
    Use the flag fast_bloom_ for a compromise between a realistic effect and the best performances
    The flag fast_bloom_ define a very fast bloom algo using only the smallest texture 
    to create a bloom effect (all the intermediate textures will be bypassed). See the bloom effect 
    project for more details.
    When fast_bloom is False, all the sub-surfaces will be blit to the final effect and will 
    produce a more realistic fire effect (this will slightly degrade the overall performances). 
    If the fire effect is too bright, you can always adjust the bright pass filter value
    bpf_threshold_(this will adjust the bloom intensity)
    bpf_threshold_ value must be in range [ 0 ... 255]   
    Below 128 the bloom effect will be more noticeable and above 128 only the brightest
    area will be enhanced.

    * LIGHT EFFECT INTENSITY

    input brightness_            : True | False
    input brightness_intensity_  : float

    When the flag is set to True, the algorithm will use an external function, 
    <shader_brightness24_exclude_inplace_c> to increase the brightness of the effect / texture
    A custom color can be passed to the function defining the pixels to be ignored during the 
    process (default is black color).
    the value must be in range [-1.0 ... 1.0]. Values below zero will decrease the brightness 
    of the flame effect and positive values will increase the brightness of the effect (causing
    bright white patches on the fire texture). 
    Values below -0.4 will cause the fire effect to be translucent and this effect can also be 
    used for simulating ascending heat convection effects on a background texture.
    
    
    * OPTIONAL SURFACE
      
    input surface_ : pygame.Surface
      
    This is an optional surface that can be passed to the shader to improve the performances 
    and to avoid a new surface to be generated every iterations. The surface size must match 
    exactly the reduce texture dimensions otherwise an exception will be raise. 
    see reduce_factor_ option to determine the fire texture size that will be processed.
    
    * COLOR PALETTE ADJUSTMENT  
    
    input adjust_palette_ : True | False
    input hsl_            : (10, 80, 1.8)

    Set this flag to True to modify the color palette of the fire texture. 
    This allow the HSL color model to be apply to the palette values
    You can redefine the palette when the flag is True and by customizing a tuple of 3 float 
    values, default is (10, 80, 1.8). 
    The first value control the palette hue value, the second is for the saturation and last, 
    the palette color lightness. 
    With the variable hsl_ you can rotate the palette colors and define a new flame
    aspect/color/intensity
    If adjust_palette_ is True the original palette define by the argument palette_, will 
    be disregarded.Instead a new palette will be created with the hsl values

    * FLAME ORIENTATION / DIRECTION & BORDER FLAME EFFECT
     
    input transpose_ = True | False,
    input border_    = True | False,
    
    transpose_ = True, this will transpose the final array 
    for e.g :  
    If the final fire texture is (w, h) after setting the transpose flag, the final 
    fire texture will become (h, w). As a result the fire effect will be transversal (starting 
    from the right of the display to the left side). 
    You can always transpose / flip the texture to get the right flame orientation  
    BORDER FLAME EFFECT 
    border_ = True to create a flame effect burning the edge of the display. This version is only
    compatible with symmetrical display or textures (same width & height). If the display 
    is asymmetric, the final border fire effect will be shown within the display and not neccessary 
    on the frame border 
    
    * FINAL TOUCH
    
    input blur_ : True | False
    
    This will will blur the fire effect for a more realistic appearance, remove all the jagged 
    edge when and pixelated effect
    
    
    :param width_           : integer; Size (width) of the surface or display in pixels
    :param height_          : integer; size (height) of the surface or display in pixels
    :param factor_          : float; Value controlling the fire height value
                              must be in range [3.95 ... 4.2].
                              The value 3.95 gives the highest flame effect
    :param palette_         : numpy.ndarray, buffer containing mapped RGB colors (uint values)
    :param fire_            : numpy.ndarray shape (w, h) containing float values (fire intensity).
                              For better performance it is advised to set the array to the size 
                              of the texture after applying the reduction_factor_.
                              For example if the reduction_factor_ is 2, the texture would have 
                              width >> 1 and height >> 1 and the fire_array should be set to 
                              numpy.empty((height >> 1, width >> 1), float32)
    :param reduce_factor_   : unsigned short int ; Can be either 0, 1, 2, 3, 4. 
                              2 and 3 provide the best performance and the best looking effect.
    :param fire_intensity_  : Integer; Control the original amount of energy at the
                              bottom of the fire, must be in range of [0 ... 32]. 
                              32 being the maximum value and the maximum fire intensity
    :param smooth_          : boolean; True smoothscale (bi-linear filtering) or
                              scale algorithm jagged edges (mush faster)
    :param bloom_           : boolean; True or False, True apply a bloom effect to the fire effect
    :param fast_bloom_      : boolean; Fastest bloom. This reduce the amount of calculation
    :param bpf_threshold_   : integer; control the bright pass filter threshold
                              value, must be in range [0 ... 255].
                              Maximum brightness amplification with threshold = 0, 
                              when bpf_threshold_ = 255, no change.
    :param low_             : integer; Starting position x for the fire effect
    :param high_            : integer; Ending position x for the fire effect
    :param brightness_      : boolean; True apply a bright filter shader to the array.
                              Increase overall brightness of the effect
    :param brightness_intensity_: float; must be in range [-1.0 ... 1.0] control
                              the brightness intensity
                              of the effect
    :param surface_         : pygame.Surface. Pass a surface to the shader for
                              better performance, otherwise a new surface will be created each 
                              calls.
    :param adjust_palette_  : boolean; True adjust the palette setting HSL
                              (hue, saturation, luminescence).
                              Be aware that if adjust_palette is True, the optional palette 
                              passed to the Shader will be disregarded
    :param hsl_             : tuple; float values of hue, saturation and luminescence.
                              Hue in range [0.0 ... 100],  saturation [0...100], 
                              luminescence [0.0 ... 2.0]
    :param transpose_       : boolean; Transpose the array (w, h) become (h, w).
                              The fire effect will start from the left and move to the right
    :param border_          : boolean; Flame effect affect the border of the texture
    :param blur_            : boolean; Blur the fire effect
    :return                 : Return a pygame surface that can be blit directly to the game display

    """
    # todo reduce_factor=0 and border = True crash

    assert reduce_factor_ in (0, 1, 2, 3, 4), \
        "Argument reduce factor must be in range 0 ... 4 " \
        "\n reduce_factor_ = 1 correspond to dividing the image size by 2" \
        "\n reduce_factor_ = 2 correspond to dividing the image size by 4"
    assert 0 <= fire_intensity_ < 33, \
        "Argument fire_intensity_ must be in range [0 ... 32] got %s" % fire_intensity_

    assert width_ > 0 and height_ > 0, "Argument width or height cannot be null or < 0"
    assert factor_ > 0, "Argument factor_ cannot be null or < 0"

    return shader_fire_effect_c(
        width_, height_, factor_, palette_, fire_,
        reduce_factor_, fire_intensity_, smooth_,
        bloom_, fast_bloom_, bpf_threshold_, low_, high_, brightness_,
        brightness_intensity_, surface_, adjust_palette_,
        hsl_, transpose_, border_, blur_
    )



@cython.binding(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tuple custom_map(int wavelength, int [:] color_array_, float gamma=1.0):
    """
    RETURN AN RGB COLOR VALUE MATCHING A CUSTOM WAVELENGTH

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    This function return a tuple (R,G,B) corresponding to the
    color wavelength define in color_array_
    (wavelength_to_rgb_custom is an External C
    routine with customize wavelength and allow the user to defined
    a customize palette according to an input value)

    example for a Fire palette
    arr = numpy.array(
        [0, 1,       # violet is not used
         0, 1,       # blue is not used
         0, 1,       # green is not used
         2, 619,     # yellow, return a yellow gradient for values [2...619]
         620, 650,   # orange return a orange gradient for values [620 ... 650]
         651, 660    # red return a red gradient for values [651 ... 660]
         ], numpy.int)


    :param wavelength   : integer; Wavelength
    :param gamma        : float; Gamma value
    :param color_array_ : numpy array containing the min and max of each color (red,
    orange, yellow, green, blue, violet)
    :return             : tuple RGB values (0 ... 255)
    """
    cdef  rgb_color_int rgb_c
    cdef int *p
    p = &color_array_[0]
    rgb_c = wavelength_to_rgb_custom(wavelength, p, gamma)
    return rgb_c.r, rgb_c.g, rgb_c.b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef fire_surface24_c(
        int width,
        int height,
        float factor,
        unsigned int [::1] palette,
        float [:, ::1] fire,
        int intensity = 0,
        int low       = 0,
        int high      = 0,
):
    """

    CREATE A FIRE EFFECT

    * Do not call this function directly

    :param width    : integer; max width of the effect
    :param height   : integer; max height of the effect
    :param factor   : float; factor to reduce the flame effect
    :param palette  : ndarray; Color palette 1d numpy array (colors buffer unsigned int values)
    :param fire     : ndarray; 2d array (x, y) (contiguous) containing float values
    :param intensity: integer; Control the flame intensity default 0 (low intensity), range [0...32]
    :param low      : integer; The x lowest position of the effect, x must be >=0 and < high
    :param high     : integer; The x highest position of the effect, x must be > low and <= high
    :return         : Return a numpy array containing the fire effect array shape
     (w, h, 3) of RGB pixels
    """

    cdef:
        # flame opacity palette
        unsigned char [:, :, ::1] out = zeros((width, height, 3), dtype=uint8)
        int x = 0, y = 0
        float d
        unsigned int ii=0
        unsigned c1 = 0, c2 = 0

    cdef int min_, max_, middle


    if low != 0 or high != 0:
        assert 0 <= low < high, "Argument low_ must be < high_"
        assert high <= width, "Argument high must be <= width"

        middle = low + ((high - low) >> 1)
        min_ = randRange(low, middle)
        max_ = randRange(middle + 1, high)
    else:
        middle = width >> 1
        min_ = randRange(0, middle)
        max_ = randRange(middle +1, width)


    with nogil:
        # POPULATE TO THE BASE OF THE FIRE (THIS WILL CONFIGURE THE FLAME ASPECT)
        for x in prange(min_, max_, schedule='static', num_threads=THREADS):
                fire[height-1, x] = randRange(intensity, 260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height-1, schedule='static', num_threads=THREADS):

            for x in range(0, width):

                    c1 = (y + 1) % height
                    c2 = x % width
                    d = (fire[c1, (x - 1 + width) % width]
                       + fire[c1, c2]
                       + fire[c1, (x + 1) % width]
                       + fire[(y + 2) % height, c2]) * factor

                    d = d - <float>(rand() * 0.0001)

                    # Cap the values
                    if d <0:
                        d = 0.0

                    # CAP THE VALUE TO 255
                    if d>255.0:
                        d = <float>255.0
                    fire[y, x] = d

                    ii = palette[<unsigned int>d % width]

                    out[x, y, 0] = (ii >> 16) & 255
                    out[x, y, 1] = (ii >> 8) & 255
                    out[x, y, 2] = ii & 255

    return asarray(out)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef inline unsigned int rgb_to_int(int red, int green, int blue)nogil:
    """
    CONVERT RGB MODEL INTO A PYTHON INTEGER EQUIVALENT TO THE FUNCTION PYGAME MAP_RGB()

    Cython cpdef function, this function can be called directly and do not require a
    hook function.

    :param red   : Red color value,  must be in range [0..255]
    :param green : Green color value, must be in range [0..255]
    :param blue  : Blue color, must be in range [0.255]
    :return      : returns a positive python integer representing the RGB values(int32)
    """
    return 65536 * red + 256 * green + blue



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline make_palette_c(int width, float fh, float fs, float fl):
    """
    CREATE A PALETTE OF RGB COLORS 

    e.g:
        # below: palette of 256 colors & surface (width=256, height=50).
        # hue * 6, saturation = 255.0, lightness * 2.0
        palette, surf = make_palette(256, 50, 6, 255, 2)
        palette, surf = make_palette(256, 50, 4, 255, 2)

    :param width  : integer, Palette width
    :param fh     : float, hue factor
    :param fs     : float, saturation factor
    :param fl     : float, lightness factor
    :return       : Return a tuple ndarray type uint32 and pygame.Surface (width, height)
    """
    assert width > 0, "Argument width should be > 0, got %s " % width

    cdef:
        unsigned int [::1] palette = numpy.empty(width, uint32)
        int x, y
        float h, s, l
        rgb rgb_

    with nogil:
        for x in prange(width, schedule='static', num_threads=THREADS):
            h, s, l = <float>x * fh,  min(fs, <float>255.0), min(<float>x * fl, <float>255.0)
            rgb_ = struct_hsl_to_rgb(h * <float>ONE_360, s * <float>ONE_255, l * <float>ONE_255)
            # build the palette (1d buffer int values)
            palette[x] = rgb_to_int(<int>(rgb_.r * <float>255.0),
                                    <int>(rgb_.g * <float>255.0),
                                    <int>(rgb_.b * <float>255.0 * <float>0.5))

    return asarray(palette)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef fire_surface24_c_border(
        int width,
        int height,

        float factor,
        unsigned int [::1] palette,
        float [:, ::1] fire,
        int intensity = 0,
        int low       = 0,
        int high      = 0,
):
    """

    CREATE A FIRE EFFECT (BORDER EFFECT)

    * Do not call this function directly

    :param width    : integer; max width of the effect
    :param height   : integer; max height of the effect
    :param factor   : float; factor to reduce the flame effect
    :param palette  : ndarray; Color palette 1d numpy array (colors buffer unsigned int values)
    :param fire     : ndarray; 2d array (x, y) (contiguous) containing float values
    :param intensity: integer; Control the flame intensity default 0 (low intensity), range [0...32]
    :param low      : integer; The x lowest position of the effect, x must be >=0 and < high
    :param high     : integer; The x highest position of the effect, x must be > low and <= high
    :return         : Return a numpy array containing the fire effect array
    shape (w, h, 3) of RGB pixels
    """

    cdef:
        # flame opacity palette
        unsigned char [:, :, ::1] out = zeros((width, height, 3), dtype=uint8)
        int x = 0, y = 0
        float d
        unsigned int ii=0
        unsigned c1 = 0, c2 = 0

    cdef int min_, max_, middle


    if low != 0 or high != 0:
        assert 0 <= low < high, "Argument low_ must be < high_"
        assert high <= width, "Argument high must be <= width"

        middle = low + ((high - low) >> 1)
        min_ = randRange(low, middle)
        max_ = randRange(middle + 1, high)
    else:
        middle = width >> 1
        min_ = randRange(0, middle)
        max_ = randRange(middle +1, width)


    with nogil:
        # POPULATE TO THE BASE OF THE FIRE (THIS WILL CONFIGURE THE FLAME ASPECT)
        # for x in prange(min_, max_, schedule='static', num_threads=THREADS
        #         fire[height - 1, x] = randRange(intensity, 260)

        # FIRE ARRAY IS [HEIGHT, WIDTH]
        for x in prange(min_, max_, schedule='static', num_threads=THREADS):
                fire[x % height, (height - 1) % width] = randRange(intensity, 260)


        # DILUTE THE FLAME EFFECT (DECREASE THE MAXIMUM INT VALUE) WHEN THE FLAME TRAVEL UPWARD
        for y in prange(1, height - 1, schedule='static', num_threads=THREADS):

            for x in range(0, width):

                    c1 = (y + 1) % height
                    c2 = x % width
                    d = (fire[c1, (x - 1 + width) % width]
                       + fire[c1, c2]
                       + fire[c1, (x + 1) % width]
                       + fire[(y + 2) % height, c2]) * factor

                    d = d - <float>(rand() * 0.0001)

                    # Cap the values
                    if d <0:
                        d = 0.0

                    if d>255.0:
                        d = <float>255.0

                    fire[x % height , y % width] = d

                    ii = palette[<unsigned int>d % width]

                    out[x, y, 0] = (ii >> 16) & 255
                    out[x, y, 1] = (ii >> 8) & 255
                    out[x, y, 2] = ii & 255

    return asarray(out)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_brightness24_exclude_inplace_c(
        unsigned char [:, :, :] rgb_array_, float shift_=0.0, color_=(0, 0, 0)):
    """
    SHADER BRIGHTNESS (EXCLUDE A SPECIFIC COLOR FROM THE PROCESS, DEFAULT BLACK COLOR)

    This shader control the pygame display brightness level
    It uses two external functions coded in C, struct_rgb_to_hsl & struct_hsl_to_rgb

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
     please refer to pygame function pixels3d or array3d to convert an image into a
     3d array (library surfarray)

    e.g:
    shader_brightness24_exclude_inplace(surface, 0.2)

    :param rgb_array_: numpy ndarray shape (w, h, 3) containing RGB pixels values
    :param shift_    : float; values in range [-1.0 ... 1.0], 0 no change,
    -1 lowest brightness, +1 highest brightness
    :param color_    : tuple; Color to excude from the brightness process
    :return          : void
    """
    assert -1.0 <= shift_ <= 1.0, \
        "Argument shift must be in range[-1.0 ... 1.0]"

    cdef Py_ssize_t width, height
    width, height = rgb_array_.shape[:2]

    cdef:
        int i=0, j=0
        unsigned char r
        unsigned char g
        unsigned char b
        float l, h, s
        hsl hsl_
        rgb rgb_
        float high, low, high_
        unsigned char rr=color_[0], gg=color_[1], bb=color_[2]

    with nogil:
        for j in range(height): #, schedule='static', num_threads=THREADS):
            for i in range(width):

                r, g, b = rgb_array_[i, j, 0], rgb_array_[i, j, 1], rgb_array_[i, j, 2]

                if not ((r==rr) and (g==gg) and (b==bb)):

                    hsl_ = struct_rgb_to_hsl(
                        r * <float>ONE_255, g * <float>ONE_255, b * <float>ONE_255)

                    l = min((hsl_.l + shift_), <float>1.0)
                    l = max(l, <float>0.0)

                    rgb_ = struct_hsl_to_rgb(hsl_.h, hsl_.s, l)

                    rgb_array_[i, j, 0] = <unsigned char> (rgb_.r * 255.0)
                    rgb_array_[i, j, 1] = <unsigned char> (rgb_.g * 255.0)
                    rgb_array_[i, j, 2] = <unsigned char> (rgb_.b * 255.0)




cdef float[5] GAUSS_KERNEL = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_blur5x5_array24_inplace_c(
        unsigned char [:, :, :] rgb_array_, mask=None):
    """
    APPLY A GAUSSIAN BLUR EFFECT TO THE GAME DISPLAY OR TO A GIVEN TEXTURE (KERNEL 5x5)

    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a 3d array (
    library surfarray)

    :param rgb_array_   : numpy.ndarray type (w, h, 3) uint8
    :param mask         : numpy.ndarray default None
    :return             : Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    """

    cdef Py_ssize_t w, h
    w, h = rgb_array_.shape[:2]

    # kernel 5x5 separable
    cdef:

        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 3), dtype=uint8)
        Py_ssize_t kernel_length = len(GAUSS_KERNEL)
        int x, y, xx, yy
        float r, g, b, s
        char kernel_offset
        unsigned char red, green, blue
        float *k
        unsigned char *c1
        unsigned char *c2
        unsigned char *c3
        unsigned char *c4
        unsigned char *c5
        unsigned char *c6

    with nogil:

        # horizontal convolution
        for y in prange(0, h, schedule='static', num_threads=THREADS):

            c1 = &rgb_array_[0, y, 0]
            c2 = &rgb_array_[0, y, 1]
            c3 = &rgb_array_[0, y, 2]
            c4 = &rgb_array_[w-1, y, 0]
            c5 = &rgb_array_[w-1, y, 1]
            c6 = &rgb_array_[w-1, y, 2]

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &GAUSS_KERNEL[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = c1[0], c2[0], c3[0]
                    elif xx > (w - 1):
                        red, green, blue = c4[0], c5[0], c6[0]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                convolve[x, y, 0] = <unsigned char>r
                convolve[x, y, 1] = <unsigned char>g
                convolve[x, y, 2] = <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule='static', num_threads=THREADS):

            c1 = &convolve[x, 0, 0]
            c2 = &convolve[x, 0, 1]
            c3 = &convolve[x, 0, 2]
            c4 = &convolve[x, h-1, 0]
            c5 = &convolve[x, h-1, 1]
            c6 = &convolve[x, h-1, 2]

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = &GAUSS_KERNEL[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = c1[0], c2[0], c3[0]
                    elif yy > (h -1):
                        red, green, blue = c4[0], c5[0], c6[0]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]
                        if red + green + blue == 0:
                            continue

                    r = r + red * k[0]
                    g = g + green * k[0]
                    b = b + blue * k[0]

                rgb_array_[x, y, 0], rgb_array_[x, y, 1], rgb_array_[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline bpf24_c(
        unsigned char [:, :, :] input_array_,
        int threshold = 128,
        bint transpose=False):
    """
    SHADER BRIGHT PASS FILTER

    Conserve only the brightest pixels in an array

    The Array (rgb_array) must be a numpy array shape (w, h, 3) containing RGB pixels,
    please refer to pygame function pixels3d or array3d to convert an image into a
    3d array (library surfarray)

    :param input_array_: numpy.ndarray shape (w, h, 3) uint8 containing RGB pixels
    :param threshold   : float Bright pass threshold default 128
    :param transpose   : bool; True| False transpose the final array
    :return            :  Return the modified array shape (w, h, 3) uint8
    """
    assert 0 <= threshold <= 255, "Argument threshold must be in range [0 ... 255]"

    cdef:
        Py_ssize_t w, h
    w, h = input_array_.shape[:2]

    cdef:
        int i = 0, j = 0
        float lum, c
        unsigned char [:, :, :] output_array_ = numpy.zeros((h, w, 3), uint8)
        unsigned char *r
        unsigned char *g
        unsigned char *b

    with nogil:
        for j in prange(0, h, schedule='static', num_threads=THREADS):
            for i in range(0, w):

                # ITU-R BT.601 luma coefficients
                r = &input_array_[i, j, 0]
                g = &input_array_[i, j, 1]
                b = &input_array_[i, j, 2]

                lum = r[0] * <float>0.299 + g[0] * <float>0.587 + b[0] * <float>0.114

                if lum > threshold:
                    c = (lum - threshold) / lum
                    output_array_[j, i, 0] = <unsigned char>(r[0] * c)
                    output_array_[j, i, 1] = <unsigned char>(g[0] * c)
                    output_array_[j, i, 2] = <unsigned char>(b[0] * c)

    return pygame.image.frombuffer(output_array_, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void shader_bloom_effect_array24_c(
        surface_,
        int threshold_,
        bint fast_ = False):
    """
    CREATE A BLOOM EFFECT

    * Surface must be a pygame Surface 24-32 bit format

    :param surface_     : pygame.Surface; Game display or texture
    :param threshold_   : integer; Threshold value uint8 in range [0 ... 255].
    The threshold value is used by a bright
     pass filter to determine the bright pixels above the given threshold.
      Below 128 the bloom effect will be more
     noticeable and above 128 a bit less.
    :param fast_        : bool; True | False; If True the bloom effect will be approximated
    and only the x16 subsurface
    will be processed to maximize the overall processing time, default is False).
    :return             : void
    """

    assert 0 <= threshold_ <= 255, "Argument threshold must be in range [0 ... 255]"

    cdef:
        Py_ssize_t  w, h
        int bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16
        bint x2, x4, x8, x16 = False

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    with nogil:
        w2, h2   = <int>w >> 1, <int>h >> 1
        w4, h4   = w2 >> 1, h2 >> 1
        w8, h8   = w4 >> 1, h4 >> 1
        w16, h16 = w8 >> 1, h8 >> 1

    with nogil:
        if w2 > 0 and h2 > 0:
            x2 = True
        else:
            x2 = False

        if w4 > 0 and h4 > 0:
            x4 = True
        else:
            x4 = False

        if w8 > 0 and h8 > 0:
            x8 = True
        else:
            x8 = False

        if w16 > 0 and h16 > 0:
            x16 = True
        else:
            x16 = False

    # SUBSURFACE DOWNSCALE CANNOT
    # BE PERFORMED AND WILL RAISE AN EXCEPTION
    if not x2:
        return

    if fast_:
        x2, x4, x8 = False, False, False

    surface_cp = bpf24_c(pixels3d(surface_), threshold=threshold_)


    # FIRST SUBSURFACE DOWNSCALE x2
    # THIS IS THE MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x2:
        s2 = scale(surface_cp, (w2, h2))
        s2_array = numpy.array(s2.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s2_array)
        # b2_blurred = frombuffer(numpy.array(s2_array.transpose(1, 0, 2),
        # order='C', copy=False), (w2, h2), 'RGB')
        b2_blurred = make_surface(s2_array)
        s2 = smoothscale(b2_blurred, (w, h))
        surface_.blit(s2, (0, 0), special_flags=BLEND_RGB_ADD)

    # SECOND SUBSURFACE DOWNSCALE x4
    # THIS IS THE SECOND MOST EXPENSIVE IN TERM OF PROCESSING TIME
    if x4:
        s4 = scale(surface_cp, (w4, h4))
        s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s4_array)
        # b4_blurred = frombuffer(numpy.array(s4_array.transpose(1, 0, 2),
        # order='C', copy=False), (w4, h4), 'RGB')
        b4_blurred = make_surface(s4_array)
        s4 = smoothscale(b4_blurred, (w, h))
        surface_.blit(s4, (0, 0), special_flags=BLEND_RGB_ADD)

    # THIRD SUBSURFACE DOWNSCALE x8
    if x8:
        s8 = scale(surface_cp, (w8, h8))
        s8_array = numpy.array(s8.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s8_array)
        # b8_blurred = frombuffer(numpy.array(s8_array.transpose(1, 0, 2),
        # order='C', copy=False), (w8, h8), 'RGB')
        b8_blurred = make_surface(s8_array)
        s8 = smoothscale(b8_blurred, (w, h))
        surface_.blit(s8, (0, 0), special_flags=BLEND_RGB_ADD)

    # FOURTH SUBSURFACE DOWNSCALE x16
    # LESS SIGNIFICANT IN TERMS OF RENDERING AND PROCESSING TIME
    if x16:
        s16 = scale(surface_cp, (w16, h16))
        s16_array = numpy.array(s16.get_view('3'), dtype=numpy.uint8)
        shader_blur5x5_array24_inplace_c(s16_array)
        # b16_blurred = frombuffer(numpy.array(s16_array.transpose(1, 0, 2),
        # order='C', copy=False), (w16, h16), 'RGB')
        b16_blurred = make_surface(s16_array)
        s16 = smoothscale(b16_blurred, (w, h))
        surface_.blit(s16, (0, 0), special_flags=BLEND_RGB_ADD)


    # if mask_ is not None:
    #     # Multiply mask surface pixels with mask values.
    #     # RGB pixels = 0 when mask value = 0.0, otherwise
    #     # modify RGB amplitude
    #     surface_cp = filtering24_c(surface_cp, mask_)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline shader_fire_effect_c(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] fire_,
        unsigned short int reduce_factor_ = 3,
        unsigned short int fire_intensity_= 32,
        bint smooth_                      = True,
        bint bloom_                       = True,
        bint fast_bloom_                  = True,
        unsigned char bpf_threshold_      = 0,
        unsigned int low_                 = 0,
        unsigned int high_                = 600,
        bint brightness_                  = True,
        float brightness_intensity_       = 0.15,
        object surface_                   = None,
        bint adjust_palette_              = False,
        tuple hsl_                        = (10, 80, 1.8),
        bint transpose_                   = False,
        bint border_                      = False,
        bint blur_                        = True
        ):
    """

    FIRE SHADER EFFECT 

    * FIRE TEXTURE SIZES 
    
    input width_  : integer,  
    input height_ : integer
    
    width_ and height_ values define the size of the texture e.g Surface(width x height)

    * FIRE ASPECT (CONTROL OVER THE WIDTH): 
    
    inputs low_ : integer  
    input high_ : integer 
    
    Optional arguments low_ & high_ (integer values) define the width 's limits of the fire effect. 
    low_ for the starting point and high_ for the ending of the effect.
    e.g low_ = 10 and high_ = 200. The fire effect will be contain within width = 10 and 200
    low_ & high_ values must be in range [0 ... width_]  
        
    * FIRE HEIGHT:
    
    input factor_ : float
    
    The fire maximum height can be adjust with the variable factor_ (float value)
    value > 3.95 will contain the effect within the display 
    value < 3.95 will enlarge the effect over the display height  
    Recommended value is 3.95 with reduce_factor_ = 3 otherwise adjust the value manually 
    to contain the fire effect within the display
        
    * SPEED CONSIDERATION
    
    input reduce_factor_ : integer
    
    The argument reduce_factor_ control the size of the texture to be processed 
    e.g : a value of 2, divide by 4 the pygame surface define by the values (width_ & height_)
    Smaller texture improve the overall performances but will slightly degrade the fire aspect, 
    especially if the blur and smooth option are not enabled.
    Recommended value for reduce_factor_ is 3 (fast process)   
    reduce_factor_ values must be an integer in range [ 0 ... 4] 
    The reduce_factor_ value will have a significant impact on the fire effect maximum height, 
    adjust the argument factor_ accordingly

    * FIRE INTENSITY AT THE SOURCE
    
    input fire_intensity_: integer
    
    Set the fire intensity with the variable fire_intensity_, 0 low flame,
    32 maximum flame effect
    Values must be an int in range [0 ... 32] 

    * SMOOTHING THE EFFECT
    
    input smooth_: True | False
    
    When smooth_ is True the algorithm will use the pygame function smoothscale (bi-linear 
    filtering) or False the final texture will be adjust with the scale function.
    Set this variable to False if you need the best performance for the effect or if you require
    a pixelated fire effect. Otherwise set the variable to True for a more realistic effect. 

    
    * BLOOM EFFECT 
    
    input bloom_         : True | False
    input fast_bloom_    : True | False
    input bpf_threshold_ : integer
       
    Fire effect produce a bright and smooth light effect to the background texture where the fire 
    intensity is at its maximum.
    Use the flag fast_bloom_ for a compromise between a realistic effect and the best performances
    The flag fast_bloom_ define a very fast bloom algo using only the smallest texture 
    to create a bloom effect (all the intermediate textures will be bypassed). See the bloom effect 
    project for more details.
    When fast_bloom is False, all the sub-surfaces will be blit to the final effect and will 
    produce a more realistic fire effect (this will slightly degrade the overall performances). 
    If the fire effect is too bright, you can always adjust the bright pass filter value
    bpf_threshold_(this will adjust the bloom intensity)
    bpf_threshold_ value must be in range [ 0 ... 255]   
    Below 128 the bloom effect will be more noticeable and above 128 only the brightest
    area will be enhanced.

    * LIGHT EFFECT INTENSITY

    input brightness_            : True | False
    input brightness_intensity_  : float

    When the flag is set to True, the algorithm will use an external function, 
    <shader_brightness24_exclude_inplace_c> to increase the brightness of the effect / texture
    A custom color can be passed to the function defining the pixels to be ignored during the 
    process (default is black color).
    the value must be in range [-1.0 ... 1.0]. Values below zero will decrease the brightness 
    of the flame effect and positive values will increase the brightness of the effect (causing
    bright white patches on the fire texture). 
    Values below -0.4 will cause the fire effect to be translucent and this effect can also be 
    used for simulating ascending heat convection effects on a background texture.
    
    
    * OPTIONAL SURFACE
      
    input surface_ : pygame.Surface
      
    This is an optional surface that can be passed to the shader to improve the performances 
    and to avoid a new surface to be generated every iterations. The surface size must match 
    exactly the reduce texture dimensions otherwise an exception will be raise. 
    see reduce_factor_ option to determine the fire texture size that will be processed.
    
    * COLOR PALETTE ADJUSTMENT  
    
    input adjust_palette_ : True | False
    input hsl_            : (10, 80, 1.8)

    Set this flag to True to modify the color palette of the fire texture. 
    This allow the HSL color model to be apply to the palette values
    You can redefine the palette when the flag is True and by customizing a tuple of 3 float 
    values, default is (10, 80, 1.8). 
    The first value control the palette hue value, the second is for the saturation and last, 
    the palette color lightness. 
    With the variable hsl_ you can rotate the palette colors and define a new flame
    aspect/color/intensity

    * FLAME ORIENTATION / DIRECTION & BORDER FLAME EFFECT
     
    input transpose_ = True | False,
    input border_    = True | False,
    
    transpose_ = True, this will transpose the final array 
    for e.g :  
    If the final fire texture is (w, h) after setting the transpose flag, the final 
    fire texture will become (h, w). As a result the fire effect will be transversal (starting 
    from the right of the display to the left side). 
    You can always transpose / flip the texture to get the right flame orientation  
    BORDER FLAME EFFECT 
    border_ = True to create a flame effect burning the edge of the display
    
    * FINAL TOUCH
    
    input blur_ : True | False
    
    This will will blur the fire effect for a more realistic appearance, remove all the jagged 
    edge when and pixelated effect
    
    
    :param width_           : integer; Size (width) of the surface or display in pixels
    :param height_          : integer; size (height) of the surface or display in pixels
    :param factor_          : float; Value controlling the fire height value
                              must be in range [3.95 ... 4.2].
                              The value 3.95 gives the highest flame effect
    :param palette_         : numpy.ndarray, buffer containing mapped RGB colors (uint values)
    :param fire_            : numpy.ndarray shape (w, h) containing float values (fire intensity).
                              For better performance it is advised to set the array to the size 
                              of the texture after applying the reduction_factor_.
                              For example if the reduction_factor_ is 2, the texture would have 
                              width >> 1 and height >> 1 and the fire_array should be set to 
                              numpy.empty((height >> 1, width >> 1), float32)
    :param reduce_factor_   : unsigned short int ; Can be either 0, 1, 2, 3, 4. 
                              2 and 3 provide the best performance and the best looking effect.
    :param fire_intensity_  : Integer; Control the original amount of energy at the
                              bottom of the fire, must be in range of [0 ... 32]. 
                              32 being the maximum value and the maximum fire intensity
    :param smooth_          : boolean; True smoothscale (bi-linear filtering) or
                              scale algorithm jagged edges (mush faster)
    :param bloom_           : boolean; True or False, True apply a bloom effect to the fire effect
    :param fast_bloom_      : boolean; Fastest bloom. This reduce the amount of calculation
    :param bpf_threshold_   : integer; control the bright pass filter threshold
                              value, must be in range [0 ... 255].
                              Maximum brightness amplification with threshold = 0, 
                              when bpf_threshold_ = 255, no change.
    :param low_             : integer; Starting position x for the fire effect
    :param high_            : integer; Ending position x for the fire effect
    :param brightness_      : boolean; True apply a bright filter shader to the array.
                              Increase overall brightness of the effect
    :param brightness_intensity_: float; must be in range [-1.0 ... 1.0] control
                              the brightness intensity
                              of the effect
    :param surface_         : pygame.Surface. Pass a surface to the shader for
                              better performance, otherwise algo is creating a new surface each 
                              calls.
    :param adjust_palette_  : boolean; True adjust the palette setting HSL
                              (hue, saturation, luminescence).
                              Be aware that if adjust_palette is True, the optional palette 
                              passed to the Shader will be disregarded
    :param hsl_             : tuple; float values of hue, saturation and luminescence.
                              Hue in range [0.0 ... 100],  saturation [0...100], 
                              luminescence [0.0 ... 2.0]
    :param transpose_       : boolean; Transpose the array (w, h) become (h, w).
                              The fire effect will start from the left and move to the right
    :param border_          : boolean; Flame effect affect the border of the texture
    :param blur_            : boolean; Blur the fire effect
    :return                 : Return a pygame surface that can be blit directly to the game display

    """


    cdef int w4, h4

    # TEXTURE DIVIDE BY POWER OF 2
    if reduce_factor_ in (0, 1, 2):
        w4, h4 = width_ >> reduce_factor_, height_ >> reduce_factor_

    # TEXTURE 150 x 150 * ratio
    elif reduce_factor_ == 3:
        # CUSTOM SIZE WIDTH 150 AND RATIO * HIGH
        w4 = 150
        h4 = <int>(150 * height_/width_)
        low_ = <int>(low_ * low_/width_)
        high_ = <int>(high_ * 150/width_)
        reduce_factor_ = 0

    # TEXTURE 100 x 100 * ratio
    elif reduce_factor_ == 4:
        w4 = 100
        h4 = <int> (100 * height_ / width_)
        low_ = <int> (low_ * low_ / width_)
        high_ = <int> (high_ * 100 / width_)
        reduce_factor_ = 0

    cdef int f_height, f_width
    f_height, f_width = (<object>fire_).shape[:2]

    assert f_width >= w4 or f_height >= h4,\
        "Fire array size mismatch the texture size.\n" \
        "Set fire_ array to numpy.empty((%s, %s), dtype=numpy.float32)" % (h4, w4)

    if surface_ is None:
        fire_surface_smallest = pygame.Surface((w4, h4)).convert()

    else:
        if PyObject_IsInstance(surface_, pygame.Surface):
            assert surface_.get_width() == w4 and surface_.get_height() == h4, \
            "Surface argument has incorrect dimension surface must be (w:%s, h:%s) got (%s, %s)\n" \
            "Set argument surface_ to None to avoid this error message"\
            % (w4, h4, surface_.get_width(), surface_.get_height())
            fire_surface_smallest = surface_
        else:
            raise ValueError("Argument surface_ must be a Surface type got %s " % type(surface_))

    if adjust_palette_:
        palette_= make_palette_c(w4, hsl_[0], hsl_[1], hsl_[2])

    if border_:
        # CREATE THE FIRE EFFECT ONTO A PYGAME SURFACE
        rgb_array_ = fire_surface24_c_border(
            w4, h4, <float>1.0 / factor_, palette_, fire_, fire_intensity_,
            low_ >> reduce_factor_, high_ >> reduce_factor_)
    else:
        rgb_array_ = fire_surface24_c(
            w4, h4, <float>1.0 / factor_, palette_, fire_, fire_intensity_,
                    low_ >> reduce_factor_, high_ >> reduce_factor_)

    # BRIGHTNESS SHADER
    if brightness_:
        # EXCLUDE BLACK COLORS (DEFAULT)
        assert -1.0 <= brightness_intensity_ <= 1.0, \
            "Argument brightness intensity must be in range [-1.0 ... 1.0]"
        shader_brightness24_exclude_inplace_c(rgb_array_=rgb_array_,
                                              shift_=brightness_intensity_, color_=(0, 0, 0))

    if blur_:
        shader_blur5x5_array24_inplace_c(rgb_array_)

    if transpose_:
        rgb_array_ = rgb_array_.transpose(1, 0, 2)
        fire_surface_smallest = rotate(fire_surface_smallest, 90)


    # CONVERT THE ARRAY INTO A PYGAME SURFACE
    array_to_surface(fire_surface_smallest, rgb_array_)


    # BLOOM SHADER EFFECT
    if bloom_:
        assert 0 <= bpf_threshold_ < 256, \
            "Argument bpf_threshold_ must be in range [0 ... 256] got %s " % bpf_threshold_
        shader_bloom_effect_array24_c(fire_surface_smallest, bpf_threshold_, fast_=fast_bloom_)

    # RESCALE THE SURFACE TO THE FULL SIZE
    if smooth_:
        fire_effect = smoothscale(fire_surface_smallest, (width_, height_))
    else:
        fire_effect = scale(fire_surface_smallest, (width_, height_))

    return fire_effect


