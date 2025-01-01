from PIL import Image, ImageDraw, ImageFont
import numpy as np

def handle_unsupported_characters(text, font, replacement='?'):
    """Attempt to handle or replace unsupported Unicode characters."""
    try:
        # Check if the text can be processed without issues
        font.getsize(text)
        return text
    except UnicodeEncodeError as e:
        # In case of an encoding error, replace unsupported characters
        processed_text = ''
        for char in text:
            try:
                font.getsize(char)
                processed_text += char
            except UnicodeEncodeError:
                processed_text += replacement
        return processed_text



# Assuming we have a sample image and a list of words for demonstration.
# Replace the 'sample_image_path' with the actual image path and 'words_list' with the actual list of words.

# Sample image path (This should be replaced with the actual image input)
sample_image_path = '/home/peter/files/FastV/src/LLaVA/images/000.png'
# Sample list of 24x24 words (This should be replaced with the actual list of words input)
words_list = [73.0, 113.0, 303.0, 27.0, 157.0, 33.0, 47.0, 204.0, 203.0, 66.0, 164.0, 107.0, 333.0, 14.0, 28.0, 137.0, 44.0, 70.0, 133.0, 207.0, 183.0, 220.0, 4.0, 40.0, 117.0, 168.0, 322.0, 80.0, 231.0, 561.0, 208.0, 247.0, 213.0, 202.0, 511.0, 294.0, 248.0, 186.0, 185.0, 173.0, 178.0, 218.0, 323.0, 152.0, 162.0, 445.0, 269.0, 369.0, 88.0, 23.0, 24.0, 31.0, 50.0, 138.0, 81.0, 67.0, 29.0, 155.0, 163.0, 205.0, 222.0, 464.0, 428.0, 212.0, 238.0, 255.0, 315.0, 417.0, 465.0, 249.0, 93.0, 421.0, 45.0, 109.0, 180.0, 131.0, 244.0, 39.0, 194.0, 292.0, 115.0, 350.0, 253.0, 15.0, 571.0, 449.0, 336.0, 198.0, 348.0, 541.0, 535.0, 287.0, 334.0, 542.0, 575.0, 491.0, 313.0, 216.0, 224.0, 129.0, 35.0, 272.0, 63.0, 343.0, 60.0, 332.0, 461.0, 324.0, 177.0, 552.0, 504.0, 301.0, 536.0, 436.0, 300.0, 497.0, 171.0, 502.0, 377.0, 489.0, 184.0, 382.0, 161.0, 416.0, 211.0, 239.0, 160.0, 153.0, 124.0, 268.0, 286.0, 285.0, 149.0, 557.0, 226.0, 550.0, 446.0, 522.0, 539.0, 547.0, 283.0, 319.0, 325.0, 111.0, 121.0, 337.0, 105.0, 258.0, 57.0, 250.0, 328.0, 518.0, 546.0, 240.0, 274.0, 281.0, 407.0, 353.0, 549.0, 54.0, 62.0, 252.0, 296.0, 562.0, 89.0, 558.0, 270.0, 430.0, 395.0, 457.0, 166.0, 467.0, 246.0, 378.0, 366.0, 229.0, 499.0, 225.0, 65.0, 423.0, 477.0, 237.0, 46.0, 10.0, 526.0, 568.0, 521.0, 340.0, 534.0, 331.0, 432.0, 104.0, 486.0, 559.0, 545.0, 422.0, 396.0, 280.0, 365.0, 373.0, 460.0, 441.0, 390.0, 399.0, 473.0, 230.0, 380.0, 189.0, 1.0, 524.0, 554.0, 143.0, 527.0, 496.0, 318.0, 523.0, 310.0, 193.0, 191.0, 172.0, 145.0, 329.0, 245.0, 278.0, 512.0, 2.0, 141.0, 217.0, 519.0, 209.0, 179.0, 388.0, 22.0, 311.0, 370.0, 381.0, 317.0, 273.0, 391.0, 515.0, 384.0, 356.0, 371.0, 260.0, 492.0, 478.0, 196.0, 306.0, 503.0, 537.0, 263.0, 484.0, 346.0, 265.0, 116.0, 309.0, 176.0, 531.0, 513.0, 442.0, 533.0, 528.0, 236.0, 84.0, 259.0, 103.0, 135.0, 312.0, 362.0, 139.0, 490.0, 235.0, 364.0, 425.0, 463.0, 393.0, 276.0, 555.0, 114.0, 372.0, 516.0, 570.0, 375.0, 374.0, 0.0, 132.0, 8.0, 267.0, 122.0, 307.0, 480.0, 409.0, 429.0, 404.0, 453.0, 564.0, 435.0, 479.0, 567.0, 357.0, 572.0, 451.0, 520.0, 158.0, 400.0, 440.0, 566.0, 405.0, 444.0, 443.0, 471.0, 525.0, 165.0, 360.0, 437.0, 482.0, 556.0, 197.0, 530.0, 439.0, 227.0, 215.0, 277.0, 406.0, 389.0, 543.0, 494.0, 288.0, 569.0, 458.0, 289.0, 424.0, 418.0, 498.0, 327.0, 383.0, 64.0, 61.0, 262.0, 495.0, 507.0, 560.0, 345.0, 434.0, 358.0, 500.0, 532.0, 517.0, 540.0, 320.0, 266.0, 112.0, 485.0, 574.0, 573.0, 563.0, 551.0, 368.0, 251.0, 408.0, 398.0, 275.0, 426.0, 565.0, 510.0, 501.0, 403.0, 548.0, 419.0, 344.0, 553.0, 233.0, 529.0, 509.0, 410.0, 330.0, 30.0, 538.0, 392.0, 361.0, 147.0, 123.0, 314.0, 100.0, 261.0, 102.0, 316.0, 79.0, 118.0, 454.0, 7.0, 206.0, 108.0, 195.0, 192.0, 174.0, 228.0, 493.0, 455.0, 242.0, 462.0, 459.0, 32.0, 363.0, 13.0, 256.0, 415.0, 420.0, 414.0, 359.0, 448.0, 69.0, 77.0, 41.0, 181.0, 232.0, 297.0, 3.0, 167.0, 367.0, 452.0, 438.0, 12.0, 5.0, 304.0, 188.0, 219.0, 385.0, 169.0, 349.0, 339.0, 130.0, 481.0, 456.0, 142.0, 487.0, 25.0, 355.0, 18.0, 214.0, 175.0, 305.0, 483.0, 200.0, 254.0, 298.0, 466.0, 291.0, 402.0, 299.0, 413.0, 401.0, 472.0, 347.0, 210.0, 187.0, 293.0, 488.0, 412.0, 505.0, 338.0, 386.0, 433.0, 508.0, 514.0, 544.0, 321.0, 341.0, 342.0, 397.0, 468.0, 21.0, 474.0, 469.0, 431.0, 387.0, 394.0, 52.0, 199.0, 351.0, 470.0, 335.0, 243.0, 376.0, 295.0, 282.0, 506.0, 427.0, 271.0, 127.0, 182.0, 190.0, 447.0, 264.0, 16.0, 170.0, 234.0, 352.0, 59.0, 150.0, 279.0, 78.0, 144.0, 257.0, 159.0, 223.0, 241.0, 354.0, 476.0, 326.0, 140.0, 120.0, 411.0, 146.0, 450.0, 151.0, 379.0, 119.0, 68.0, 55.0, 302.0, 134.0, 201.0, 101.0, 71.0, 110.0, 95.0, 284.0, 97.0, 99.0, 475.0, 126.0, 148.0, 156.0, 9.0, 106.0, 308.0, 128.0, 125.0, 154.0, 56.0, 221.0, 92.0, 136.0, 290.0, 94.0, 38.0, 43.0, 51.0, 36.0, 83.0, 49.0, 86.0, 91.0, 20.0, 90.0, 74.0, 82.0, 87.0, 98.0, 85.0, 96.0, 48.0, 72.0, 75.0, 76.0, 19.0, 58.0, 53.0, 37.0, 17.0, 26.0, 42.0, 34.0, 11.0, 6.0]

# words_list = words_list[0].split()
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
        word = str(word)
        # Calculate text size and position
        text_width, text_height = draw.textsize(word, font=font)
        text_x = x + (cell_width - text_width) / 2
        text_y = y + (cell_height - text_height) / 2
        
        # Draw the word
        safe_word = handle_unsupported_characters(word, font)
        text_width, text_height = draw.textsize(safe_word, font=font)
        draw.text((text_x, text_y), safe_word, fill=tex_color, font=font)
        

# save the image
output_image.save('/home/peter/files/LLaVA/my_QA/image/rank2.jpg')
# Note: This code assumes that the PIL library is installed and a valid font path is provided.
# You may need to adjust the font path and size according to your setup and requirements.

data = np.reshape(words_list, (24, 24), order='C')

normalized_data = data / np.max(data)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[1].imshow(normalized_data, cmap='hot')
plt.savefig('/home/peter/files/LLaVA/my_QA/image/heatmap6.png')