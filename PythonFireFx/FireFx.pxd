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

cdef extern from 'Include/Shaderlib.c':

    struct hsl:
        float h
        float s
        float l

    struct rgb:
        float r
        float g
        float b

    struct rgb_color_int:
        int r;
        int g;
        int b;

    struct extremum:
        int min;
        int max;

    hsl struct_rgb_to_hsl(float r, float g, float b)nogil;
    rgb struct_hsl_to_rgb(float h, float s, float l)nogil;
    rgb_color_int wavelength_to_rgb(int wavelength, float gamma)nogil;
    rgb_color_int wavelength_to_rgb_custom(int wavelength, int arr[], float gamma)nogil;
    float randRangeFloat(float lower, float upper)nogil;
    int randRange(int lower, int upper)nogil;



cpdef shader_fire_effect(
        int width_,
        int height_,
        float factor_,
        unsigned int [::1] palette_,
        float [:, ::1] fire_,

        # OPTIONAL
        unsigned short int reduce_factor_ = *,
        unsigned short int fire_intensity_= *,
        bint smooth_                      = *,
        bint bloom_                       = *,
        bint fast_bloom_                  = *,
        unsigned char bpf_threshold_      = *,
        unsigned int low_                 = *,
        unsigned int high_                = *,
        bint brightness_                  = *,
        float brightness_intensity_       = *,
        object surface_                   = *,
        bint adjust_palette_              = *,
        tuple hsl_                        = *,
        bint transpose_                   = *,
        bint border_                      = *,
        bint blur_                        = *
        )