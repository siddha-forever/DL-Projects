from roboflow import Roboflow

rf=Roboflow(api_key="dlURMCVFA9rzEFf6fQjn")

project = rf.workspace().project("fire-d6yfv")

model=project.version(1).model #yolov9 model

model.predict("images/fire.450.png", confidence=20).save("result.jpg")
print("file saved")