#Source: https://stackoverflow.com/questions/56920338/some-time-program-works-fine-and-some-times-shows-height-width-layers-img-sh
img_array = []
for filename in glob.glob('/home/adnan/Downloads/*.png'):
    img = cv2.imread(filename)
    print(img)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()