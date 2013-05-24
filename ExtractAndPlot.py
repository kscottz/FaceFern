from SimpleCV import Image, ImageSet, Color, Display
import matplotlib.pyplot as plt
import numpy as np
nDesc = 64
imgsz = 255
x1 = np.round(imgsz*np.random.rand(1,nDesc))[0].tolist()
x2 = np.round(imgsz*np.random.rand(1,nDesc))[0].tolist()
y1 = np.round(imgsz*np.random.rand(1,nDesc))[0].tolist()
y2 = np.round(imgsz*np.random.rand(1,nDesc))[0].tolist()
p1 = zip(x1,y1)
p2 = zip(x2,y2)
descriptors = zip(p1,p2)

def extract(iset,nDesc,descriptors):
    posCounts = np.zeros([1,nDesc])[0]
    for face in iset:
        for idx, d in enumerate(descriptors):
            if( face[d[0][0],d[0][1]][0] > face[d[1][0],d[1][1]][0] ):
                posCounts[idx] += 1
    return np.array(posCounts)/len(iset)

faces = ImageSet('./faces/')
faceResult = extract(faces,nDesc,descriptors)
print faceResult
notFaces = ImageSet('./negs/')
negResult = extract(notFaces,nDesc,descriptors)
print negResult
fig = plt.figure()
ax = fig.add_subplot(111)
ind = np.arange(nDesc) 
width = 0.5
rects1 = ax.bar(ind, faceResult, width, color='r')
rects2 = ax.bar(ind+width, negResult, width, color='b')
# add some
ax.set_ylabel('Percentage with descriptor == 1')
ax.set_title('Random Binary Descriptor Incidence')
ax.legend( (rects1[0], rects2[0]), ('faces', 'notfaces') )

plt.show() 
