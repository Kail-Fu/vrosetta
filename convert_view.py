from bs4 import BeautifulSoup
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.spatial import distance

pics = []
for i in range(1, 61):
     og = i
     webpage = "view"+str(i)+".html"
     if i<10:
        i = "00"+str(i)
     elif (i<100):
        i = "0"+str(i)
     else:
        i = str(i)


     image_file = "../test_images/OGER_Page_"+str(i)+".jpg"
     image = Image.open(image_file)
     ######
     old_size = image.size
     new_size = (800, 800)
     new_im = Image.new("RGB", new_size)
     new_im.paste(image, ((new_size[0]-old_size[0])//2,
                      (new_size[1]-old_size[1])//2))
     new_im.show()
    ######


     with open(os.path.join(sys.path[0], webpage), "r") as f:
        html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        words = soup.find_all('span', attrs={'class':'fn-note-content'})
        locs = soup.findAll("div", {"class": "fn-area"})
        locations = []
        for loc in locs:
            locations.append(loc['style'])
        old_size = image.size

        new_size = (old_size[0], int(old_size[1]*(2)))
        new_im = Image.new("RGB", new_size, 'white')
        new_im.paste(image, ((0, 0)))
        draw = ImageDraw.Draw(new_im)
        font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', size=30)
        color = 'rgb(0, 0, 0)' # black color
        #old_im = Image.open('someimage.jpg')

        #pos_x = 390
        #pos_y = 85
        #for j in range(0, 11):
        position = []
        pos_to_sort = []
        for j in range(len(words)):
            left = int(locations[j].split()[1][:-3])
            top = int(locations[j].split()[3][:-3])
            height = int(locations[j].split()[7][:-3])
            position.append((left, top))
            pos_to_sort.append((distance.euclidean((left, top), (0, 0))))
            #zero_mat.append((0, 0))
            #distances = np.append(distances, [(left, top+height)])
            #distances[j] = [[left, top+height]]

        dist = distance.cdist(position, position, 'euclidean')
        #pos_sort = distance.cdist(position, zero_mat, 'euclidean')

        start = np.argsort(pos_to_sort)[0]
        #print(np.argsort(pos_to_sort))
        words_sorted = []
        pos_sorted = []
        already_sorted = [start]
        for j in range(len(words)):
            #print(start)
            words_sorted.append(words[int(start)])
            pos_sorted.append((int(locations[start].split()[1][:-3]), int(locations[start].split()[3][:-3]), int(locations[start].split()[7][:-3])))
            if(j<len(words)-1):
                count = 1
                new_start = np.argsort(dist[start])[count]
                while((len(list(filter (lambda x : x == new_start, already_sorted))) > 0)):
                    count+=1
                    new_start = np.argsort(dist[start])[count]
                start = new_start
                already_sorted.append(start)
        #print(sorted)
        #print(width)
        #print(pos_sorted)
        text = str("")
        #f= open("caption"+str(i)+".txt","w+")
        for j in range(len(words)):
            left = pos_sorted[j][0]
            top = pos_sorted[j][1]
            height = pos_sorted[j][2]
            pos_x=left

            pos_y = top+height

            #f.write((str(og)+"."+str(j+1)+") "+words_sorted[j].get_text()).encode('utf-8')+'\n')
            temp = (words_sorted[j].get_text()).replace('\n', ' ')
            text = text + " ("+str(og)+"."+str(j+1)+") "+ temp
            #if(j<8):
            #    pos_y=top+height*1.9
            #else:
            #    pos_y=top+height*2.3
            #(x, y) = (left, top)
            #print(words[j].text)
            #message = words[j].text
            #name = words[j].text
            #draw.text((700*1.875, 425*1.875), str(j+1), font=font, color=color)
            draw.text((pos_x*1.875, pos_y*1.82), str(j+1), font=font, color=color)
        #wraps text around image so it fits on the image

        font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', size=30)
        max_width = 1500
        lines = []
        if font.getsize(text)[0]  <= max_width:
            lines.append(text)
        else:
            #split the line by spaces to get words
            words = text.split(' ')
            k = 0
            # append every word to a line while its width is shorter than the image width
            while k < len(words):
                line = ''

                while k < len(words) and font.getsize(line + words[k])[0] <= max_width:
                    line = line + words[k]+ " "
                    k += 1
                if not line:
                    line = words[k]
                    k += 1
                lines.append(line)

        x = 10
        y=old_size[1]
        for line in lines:
            draw.text((x, y), line, fill=(0, 0, 0), font=font)
            y = y+40
        #draw.text((10, 1000), lines, font=font, fill=(0, 0, 0))
            #draw.text((x, y), name, fill=color, font=font)


        #new_im.save('annotated_image'+str(i)+'.png')
        pics.append(new_im)
images = pics
widths, heights = zip(*(i.size for i in images))



total_width = sum(widths)
max_height = max(heights)
max_width = max(widths)
new_im = Image.new('RGB', (max_width*50, max_height*3))

x_offset = 0
count = 0
y_offset = 0
for j in range(60):
  count+=1
  new_im.paste(images[j], (x_offset, y_offset))
  if(count==20):
      x_offset=0
      y_offset += max_height
      count=0
  else:
      x_offset += images[j].size[0]

#print("here")
#new_im.show()
print("there")
new_im.save('20by3_wall1_1.png')

#row = 7
#dst = Image.new('RGB', (new_im.width, new_im.height * row))
#for y in range(row):
#    dst.paste(new_im, (0, y * new_im.height))
#dst.save("full_image.jpg")

#im1 = pics[0]
#im2=pics[1]
#im3=pics[2]
#im4 = pics[3]
#dst = Image.new('RGB', (im1.width + im2.width, im1.height +im2.height))
#dst.paste(im1, (0, 0))
#dst.paste(im2, (im1.width, 0))
#dst.paste(im3, (0, im1.height))
#dst.paste(im4, (im1.width, im1.height))
#dst.save('tiled_im.png')
