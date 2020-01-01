// Original Source Code by Meroni (https://github.com/Flowx08/).
// Modified by Curtó & Zarza.
// c@decurto.ch z@dezarza.ch

#include "TextRendering.hpp"
#include "font8x8_basic.h"
#include <string.h>

void render_char(float* image_data, const uint image_width, const uint image_height, const uint image_channels,
		char *char_bitmap, const uint x, const uint y, const uint size, long color)
{
	int set;
	int mask;
	float scale = 8.f / (float)size;
	#define GETBYTE(n, byte_pos) (n >> (8 * byte_pos)) & 0xFF
	for (uint z = 0; z < size; z++) {
		if (x + z >= image_width) continue;
		for (uint ct = 0; ct < size; ct++) {
			if (y + ct >= image_height) continue;
			set = char_bitmap[(int)(ct * scale)] & 1 << (int)(z * scale);
			if (set) {
				for (int c = 0; c < image_channels; c++) {
					image_data[((y + ct) * image_width + x + z) * image_channels + c] = GETBYTE(color, (image_channels -1 -c));
				}
			}
		}
	}
}

void text_draw(float* image_data, const uint image_width, const uint image_height, const uint image_channels,
	const char* text, const uint x, const uint y, const uint char_size, const long color)
{
	const int padding = char_size / 4.f;
	const int text_lengt = strlen(text);
	for (int z = 0; z < text_lengt; z++)
		render_char(image_data, image_width, image_height, image_channels, font8x8_basic[(int)text[z]],
				x + z * (char_size + padding), y, char_size, color);		
}
