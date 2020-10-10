import os

imageFiles = [f for f in os.listdir('.') if (os.path.isfile(f) and ".py" not in f)]
i = 1
for f in imageFiles:
    os.rename(f, "test"+str(i)+".jpg")
    i+=1
