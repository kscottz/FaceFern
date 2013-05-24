from SimpleCV import Image, Display, ImageSet, Color, ROI
import numpy as np
import os as os
sz = 256
count = 0
outdir = './faces/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
negTotal = 0
negcount = 5
negdir = './negs/'
if not os.path.exists(negdir):
    os.mkdir(negdir)

disp = Display()
iset = ImageSet('data')
for img in iset:
    img.save(disp)
    face = img.findHaarFeatures('face.xml')    
    if( face is not None and len(face) > 0 ):
        face[-1].draw(width=3)
        img.applyLayers().save(disp)
        fname = outdir + 'Face' + str(count)+'.png'
        face[-1].crop().resize(sz,sz).toGray().save(fname)
        print "Saving: " + fname
        count = count + 1
        ncount = 0
        while ncount < negcount:
            pt = 256*np.random.rand(1,2)
            roi = ROI(np.round((img.width-sz)*np.random.rand(1))[0],np.round((img.height-sz)*np.random.rand(1))[0],sz,sz,img)
            #print roi.toXYWH()
            if( not face[-1].overlaps(roi) ):
                #roi.draw(width=3)
                #img.applyLayers().save(disp)
                fname = negdir+"Neg"+str(negTotal)+".png"
                roi.crop().toGray().save(fname)
                ncount += 1
                negTotal += 1 
