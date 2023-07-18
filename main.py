import numpy as np
import cv2 


proto_pth = 'models/colorization_deploy_v2.prototxt' 
model_pth = 'models/colorization_release_v2.caffemodel'
kernel_pth = 'models/pts_in_hull.npy'
img_pth = 'ten.jpg'

net= cv2.dnn.readNetFromCaffe(proto_pth, model_pth)

#cluster center points of kernels
points=np.load(kernel_pth)

#reshaping to make it a convolutional kernel with 1,1 size 
points=points.transpose().reshape(2,313,1,1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313],2.606, dtype="float32")]


bw_img = cv2.imread(img_pth)
normalized = bw_img.astype("float32")/255.0

LAB=cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
LAB=cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB) #bgr instead of rgb because imread loads the image in bgr instead of rgb (Ask oepnCV idk)

#resizing image because the model only works on 224*224 pixels (model is trained on these dimensions)
resize = cv2.resize(LAB,(224,224))
L = cv2.split(resize)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab=net.forward()[0, :, :, :].transpose((1,2,0))

ab = cv2.resize(ab, (bw_img.shape[1], bw_img.shape[0]))
L = cv2.split(LAB)[0]

colorized = np.concatenate((L[:,:,np.newaxis],ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)

colorized = (colorized * 255.0).astype("uint8")

cv2.imshow("Black and White image",bw_img)
cv2.imshow("Colorized",colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()







