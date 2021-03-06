from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os

'''
This class implements a meme drawer, where the meme image is initialized and a text can be written on top.

Methods:
- save: saves the image
- get_font_size: calculates the font size
- write_text: writes the text on the image
- write_text_box: splits the text to lines according to box_width and uses write_text to write each line
- get_text_size: returns the sixe of the text
'''


class ImageText(object):
    def __init__(self, filename_or_size, desired_size=300, mode='RGBA', background=(0, 0, 0, 0),
                 encoding='utf8'):
        if filename_or_size is None:
            raise ValueError('filename_or_size is None!')
        if isinstance(filename_or_size, str):
            self.filename = filename_or_size
            image = Image.open(self.filename)
            baseheight = desired_size
            hpercent = (baseheight/float(image.size[1]))
            wsize = int((float(image.size[0])*float(hpercent)))
            self.image = image.resize((wsize, baseheight), Image.ANTIALIAS)
            self.size = self.image.size
        elif isinstance(filename_or_size, (list, tuple)):
            self.size = filename_or_size
            self.image = Image.new(mode, self.size, color=background)
            self.filename = None
        elif isinstance(filename_or_size, Image.Image):
            self.filename = filename_or_size
            image = filename_or_size
            baseheight = desired_size
            hpercent = (baseheight/float(image.size[1]))
            wsize = int((float(image.size[0])*float(hpercent)))
            self.image = image.resize((wsize, baseheight), Image.ANTIALIAS)
            self.size = self.image.size
        self.draw = ImageDraw.Draw(self.image)

    def save(self, filename=None):
        self.image.save(filename or self.filename)

    def get_font_size(self, text, font, max_width=None, max_height=None):
        if max_width is None and max_height is None:
            raise ValueError('You need to pass max_width or max_height')
        font_size = 1
        text_size = self.get_text_size(font, font_size, text)
        if (max_width is not None and text_size[0] > max_width) or \
           (max_height is not None and text_size[1] > max_height):
            raise ValueError("Text can't be filled in only (%dpx, %dpx)" %
                             text_size)
        while True:
            if (max_width is not None and text_size[0] >= max_width) or \
               (max_height is not None and text_size[1] >= max_height):
                return font_size - 1
            font_size += 1
            text_size = self.get_text_size(font, font_size, text)

    def write_text(self, x, y, text, font_filename, font_size=11,
                   color=(0, 0, 0), max_width=None, max_height=None, 
                   stroke_width=2, stroke_fill=(0, 0, 0)):
        if font_size == 'fill' and \
           (max_width is not None or max_height is not None):
            font_size = self.get_font_size(text, font_filename, max_width,
                                           max_height)
        text_size = self.get_text_size(font_filename, font_size, text)
        font = ImageFont.truetype(font_filename, font_size)
        if x == 'center':
            x = (self.size[0] - text_size[0]) / 2
        # if y == 'center':
            #y = (self.size[1] - text_size[1]) / 2
        self.draw.text((x, y), text, font=font, fill=color,
                       stroke_width=stroke_width, stroke_fill=stroke_fill)
        return text_size

    def get_text_size(self, font_filename, font_size, text):
        font = ImageFont.truetype(font_filename, font_size)
        return font.getsize(text)

    # write_text_box will split the text in many lines, based on box_width
    # `place` can be 'left' (default), 'right', 'center' or 'justify'
    # write_text_box will return (box_width, box_calculed_height) so you can
    # know the size of the wrote text
    def write_text_box(self, x, y, text, box_width, font_filename=os.path.join("font", "impact.ttf"),
                       font_size=11, color=(0, 0, 0), place='left',
                       justify_last_line=False, bottom=False):
        lines = [1, 2, 3]
        words = text.split()
        while len(lines) >= 3:
            line = []
            lines = []
            for word in words:
                new_line = ' '.join(line + [word])
                size = self.get_text_size(font_filename, font_size, new_line)
                text_height = size[1]
                if size[0] <= box_width:
                    line.append(word)
                else:
                    lines.append(line)
                    line = [word]
            if line:
                lines.append(line)
            lines = [' '.join(line) for line in lines if line]
            if len(lines) >= 3:
                font_size -= 3

        if bottom and len(lines) > 1:
            y -= font_size

        height = y
        for index, line in enumerate(lines):
            if index != 0:
                height += text_height
            if place == 'left':
                self.write_text(x, height, line, font_filename, font_size,
                                color)
            elif place == 'right':
                total_size = self.get_text_size(font_filename, font_size, line)
                x_left = x + box_width - total_size[0]
                self.write_text(x_left, height, line, font_filename,
                                font_size, color)
            elif place == 'center':
                total_size = self.get_text_size(font_filename, font_size, line)
                x_left = int(x + ((box_width - total_size[0]) / 2))
                self.write_text(x_left, height, line, font_filename,
                                font_size, color)
            elif place == 'justify':
                words = line.split()
                if (index == len(lines) - 1 and not justify_last_line) or \
                   len(words) == 1:
                    self.write_text(x, height, line, font_filename, font_size,
                                    color)
                    continue
                line_without_spaces = ''.join(words)
                total_size = self.get_text_size(font_filename, font_size,
                                                line_without_spaces)
                space_width = (box_width - total_size[0]) / (len(words) - 1.0)
                start_x = x
                for word in words[:-1]:
                    self.write_text(start_x, height, word, font_filename,
                                    font_size, color)
                    word_size = self.get_text_size(font_filename, font_size,
                                                   word)
                    start_x += word_size[0] + space_width
                last_word_size = self.get_text_size(font_filename, font_size,
                                                    words[-1])
                last_word_x = x + box_width - last_word_size[0]
                self.write_text(last_word_x, height, words[-1], font_filename,
                                font_size, color)
        return (box_width, height - y)
    