def detectV1(image):
    faces = cas.detectMultiScale(image, 1.1,4)
    
    for (x,y,w,h) in faces:
        cv.rectangle(image, (x,y),(x+w,y+h), (0,0,255),2)
        
    crops = []
    for (x,y,w,h) in faces:
        new_img = cv.resize(image[y:y+h, x:x+w], (50,50))
        crops.append(new_img)

    classes = ['WithMask','WithoutMask']

    ans = []

    for x in crops:
        ans.append(classes[int(np.round(model.predict(np.expand_dims(crops[0],0))[0][0]))])
        #uncomment for plotting single images
        #plt.imshow(x)
        #plt.show()

    return ans        
