import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import project3
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

def array2PIL(data):
  mode = 'RGB'
  size = data.shape[1],data.shape[0]
  return Image.frombuffer(mode, size, data.tostring(), 'raw', mode, 0, 1)

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
angles = None
prev_image = 0
cnt=0
debug=1
fo = None

@sio.on('telemetry')
def telemetry(sid, data):
    global cnt
    global angles
    global fo

    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    #print ("shape = ",image_array.shape)
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    #steering_angle=angles[cnt]

    cnt+=1
    #capture before data

    steering_angle, throttle, combined_image = project3.cvangle(cnt, image_array, float(throttle), float(speed))
    #capture result angle
    output = str(cnt) + "," + data["steering_angle"] + "," + \
        data["throttle"] + "," + data["speed"] + "," + str(steering_angle)+  \
        "," + str(throttle)

    fo.write(output + "\n")

    print("steering,",output)
    #print("steering ",cnt, np.round(steering_angle,2), throttle, np.round(float(speed),1))

    #img2 = array2PIL(combined_image)
    #img2.save('./out/'+ str(cnt) + '.jpg')

    fig = plt.figure()
    plt.title('angle = ' + str(steering_angle))
    plt.imshow(combined_image)
    plt.savefig('./out/'+ str(cnt) + '.jpg')
    plt.close(fig)
    img1 = array2PIL(image_array)
    img1.save('./input/'+ str(cnt) + '.jpg')

    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    global fo

    print("connect ", sid)
    cnt = 0
    if(fo is not None):
        fo.close()
    fo = open("./output.log", "w+")
    send_control(0, 0)

#@sio.on('disconnect')
#def test_disconnect():
#    print('Client disconnected')
#    fo.close()

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')

    #angles = np.loadtxt('/Users/sjamthe/Downloads/steer.txt')
    cnt = 0
    #fo.write("file opened")
    #parser.add_argument('model', type=str,
    #help='Path to model definition json. Model weights should be on the same path.')
    #args = parser.parse_args()
    #with open(args.model, 'r') as jfile:
     #   model = model_from_json(json.load(jfile))

    #model.compile("adam", "mse")
    #weights_file = args.model.replace('json', 'h5')
    #model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
