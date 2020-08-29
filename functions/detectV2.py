def detectV2(image):
    faces = cas.detectMultiScale(image, 1.1,4)
    font = cv.FONT_HERSHEY_PLAIN 
    color = (255, 255, 255)
    classes = ['WithMask','WithoutMask']
    for (x,y,w,h) in faces:
        cv.rectangle(image, (x,y),(x+w,y+h), (0,0,255),2)
        new_img = cv.resize(image[y:y+h, x:x+w], (50,50))
        pred = model.predict(np.expand_dims(new_img,0))[0][0]
        cv.rectangle(image, (x, y+h),(x+w,y+h+30) , (0,0,255), thickness=-1)
        cv.putText(image, classes[int(np.round(pred))]  , (x+5,y+h+25), font, 2, color,2)
    return image
