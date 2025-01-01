from PIL import Image, ImageDraw, ImageFont
import numpy as np

sample_image_path = '/home/peter/files/FastV/src/LLaVA/images/024.png'
# sample_image_path = '/home/peter/files/LLaVA/my_QA/image/003.jpg'
id = sample_image_path.split('/')[-1].split('.')[0]
# Sample list of 24x24 words (This should be replaced with the actual list of words input)
words_list = ['colored', '星', '्', 'izon', 'ello', 'seen', 'al', 'GS', 'fty', 'ло', 'GS', 'Farm', 'dying', 'icum', 'HS', '्', '�', '्', 'место', '्', '्', 'HS', 'HS', 'ν', 'itime', '्', 'seen', 'font', 'Bruno', 'HS', 'flowers', '位', 'play', 'ء', '�', 'ская', 'far', '�', 'hl', '�', '्', '्', 'HS', 'hl', '्', 'cy', '�', 'dale', 'Or', '्', 'al', '/', '्', '्', 'dale', '्', '位', '्', '*', 'asures', 'actory', '्', 'HS', '्', 'trees', 'HS', 'hl', 'hl', '्', '्', 'hl', 'dale', 'And', 'al', '�', '्', '्', 'HS', '्', 'y', 'HS', '्', '्', 'Why', '्', '्', '्', '्', '्', 'HS', '्', 'aine', '्', '्', '्', 'dy', '्', 'Who', 'al', '्', 'ло', '्', 'aders', 'RO', '्', 'IGN', '्', '्', '्', '�', '्', '्', '�', '्', '्', 'dale', '्', 'aine', 'arta', 'green', 'ifying', 'al', '�', '्', 'ская', '्', '্', 'IGN', 'HS', '्', '्', '्', '्', 'dale', '्', 'al', '्', '्', 'igh', '्', '्', '्', '्', '्', 'owned', 'owned', 'dale', '्', 'ething', 'gg', '्', 'HS', '्', 'aine', '�', '्', 'asures', '्', '्', '्', '्', 'HS', '्', '�', '्', '्', 'aine', '्', 'itime', 'al', 'My', 'seen', '्', 'utter', '्', '्', '星', '्', '्', 'actory', 'HS', '्', '्', '्', 'far', 'hl', 'igh', '्', 'dale', '्', '/', 'aine', 'iful', 'dy', 'itime', '्', '्', '्', '्', '्', '्', '्', 'cm', 'Cont', '起', 'lady', 'lady', '親', 'ize', 'val', '्', '्', '्', 'HS', '्', 'gg', 'al', 'uman', 'far', 'dale', '्', 'front', '्', 'hl', 'jav', 'uns', 'DATE', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', '्', 'val', 'DATE', 'itime', 'hl', '्', '्', '्', 'font', '�', 'Team', 'ская', 'Cat', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'HS', 'vy', 'ן', 'dy', 'HS', '/', '्', 'actory', '्', 'Tim', 'sl', 'DATE', 'white', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', '्', 'itime', 'itime', 'On', 'itime', 'roads', 'Tim', '�', '्', 'puzz', 'sl', 'ская', 'far', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', '�', 'FS', 'Tim', 'dale', 'dale', 'dale', 'pp', 'dale', '्', 'круп', 'dale', 'iles', 'ская', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'dale', 'dale', 'do', '्', 'do', 'FS', '्', '्', 'gem', 'do', 'phon', 'dale', 'iles', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'dale', 'ヤ', '�', 'ft', 'ARD', 'ская', 'FS', '्', '्', 'ч', 'dale', 'dale', 'dale', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'do', 'encer', 'FS', 'ft', 'ke', 'pivot', 'sl', '忠', '्', 'sl', '्', 'far', 'ard', 'FS', 'FS', 'sl', 'den', 'sl', 'far', 'Rest', 'Rest', '起', '्', '्', 'do', 'dale', 'ann', 'ская', 'ke', 'far', 'far', 'HS', 'seen', 'ty', '�', '�', 'iful', 'itime', 'Canal', 'ive', 'FS', 'ская', 'thur', 'FS', 'FS', 'RO', 'thur', '्', 'phon', 'possibly', 'young', 'young', 'under', 'Α', 'seen', 'al', 'utter', 'ty', '्', 'edit', 'QL', 'ы', '�', 'FS', 'Canal', '्', 'ibly', 'FS', 'FS', 'ز', 'owy', 'кти', 'cks', 'ft', 'ская', 'ke', 'ская', 'hl', 'ская', 'ty', '्', 'actory', 'ething', 'ская', '�', '्', 'ugly', '्', 'al', '्', 'ize', 'aine', 'ская', '्', 'al', '्', '्', 'itime', 'far', '्', 'On', '्', 'al', 'сті', '्', '्', 'aine', 'seen', '्', 'ty', 'mathfrak', 'actory', 'HS', '्', 'IS', 'far', 'far', 'CON', '्', '्', '्', 'itime', 'val', 'asy', '्', '्', '्', 'ская', 'HS', '्', '्', '्', '�', 'час', 'igh', '्', 'supposed', '�', 'supposed', '्', 'hl', 'gg', 'val', '्', '์', 'aine', 'HS', 'arta', 'itime', 'agua', 'agua', '`,', '�', 'agua', 'agua', 'dale', 'al', '्', '्', 'cy', 'hl', 'aine', '�', '�', '्', 'HS', 'flowers', 'aine', 'aine', 'пу', 'My', 'HS', 'HS', 'itime', 'ν', 'hosted', '्', 'aine', '/', 'HS', '्', 'iful', '्', '्', 'grey', '्', 'hl', 'hl', '्', '्', '्', 'itime', 'aine', 'iful']


print(len(words_list))

# words_list = words_list[0].split()
print(len(words_list))
# Load the image
image = Image.open(sample_image_path)

# Resize the image to 2100x2100
image = image.resize((2100, 2100))

# Create a new image for drawing
output_image = Image.new('RGB', (2100, 2100), (255, 255, 255))
output_image.paste(image, (0, 0))

# Create a drawing context
draw = ImageDraw.Draw(output_image)

# Define grid size
grid_size = 24

# Calculate cell size
cell_width = 2100 // grid_size
cell_height = 2100 // grid_size

# Load a font
font_size = 20
font = ImageFont.truetype("Roboto/Roboto-Black.ttf", font_size)
tex_color = (255, 0, 0)


# Iterate over the grid
for i in range(grid_size):
    for j in range(grid_size):
        # Calculate cell position
        x = j * cell_width
        y = i * cell_height
        
        # Draw cell boundaries
        draw.rectangle([x, y, x+cell_width, y+cell_height], outline=(0, 0, 0))
        
        # Get the corresponding word
        word = words_list[i * grid_size + j]
        # if not (word == 'piano' or word == 'board' or word == 'white' or word == 'black' or word == 'keyboard'):
        #     word = ''
        word = str(word)
        # Calculate text size and position
        # text_width, text_height = draw.textsize(word, font=font)
        # text_x = x + (cell_width - text_width) / 2
        # text_y = y + (cell_height - text_height) / 2
        bbox = font.getbbox(word)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = x + (cell_width - text_width) / 2
        text_y = y + (cell_height - text_height) / 2
        
        # Draw the word
        # safe_word = handle_unsupported_characters(word, font)
        # text_width, text_height = draw.textsize(safe_word, font=font)
        draw.text((text_x, text_y), word, fill=tex_color, font=font)
        

# save the image
save_path = 'look.jpg'
output_image.save(save_path)
 