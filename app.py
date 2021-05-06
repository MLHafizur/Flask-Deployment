import queue
from sys import stdout
import logging
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from camera import Camera
from utils import base64_to_pil_image, pil_image_to_base64
import cv2
import numpy as np
from process import webopencv
from queue import Queue


#----------------- Video Transmission ------------------------------#
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['DEBUG'] = True
socketio = SocketIO(app)

cam = cv2.VideoCapture(0)
@app.route("/")
def home():
    """The home page with webcam."""
    return render_template('video_test.html')


@app.route("/view")
def view():
    """The client page."""
    return render_template('client.html')

#---------------- Video Socket Connections --------------------------#
@socketio.on('connect', namespace='/live')
def test_connect():
    """Connect event."""
    print('Client wants to connect.')
    emit('response', {'data': 'OK'})


@socketio.on('disconnect', namespace='/live')
def test_disconnect():
    """Disconnect event."""
    print('Client disconnected')


@socketio.on('event', namespace='/live')
def test_message(message):
    """Simple websocket echo."""
    emit('response',
         {'data': message['data']})
    print(message['data'])


@socketio.on('livevideo', namespace='/live')
def test_live(message):
    """Video stream reader."""
    app.queue.put(message['data'])



with open(r"C:\Users\hafizurr\Documents\test\backup\classes.names") as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet(r"C:\Users\hafizurr\Documents\test\backup\yolovhafiz.cfg",
                                    r"C:\Users\hafizurr\Documents\test\backup\backup\yolovhafiz_best.weights")

layers_names_all = network.getLayerNames()

layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]


probability_minimum = 0.60

threshold = 0.60


colours = np.array([[0, 255, 0], [0, 0, 255], [255, 0, 0]], np.uint8)



def gen_frames():

    while True:
        _, frame = cam.read()

        h, w = None, None


        if w is None or h is None:
            h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

        network.setInput(blob)  
        output_from_network = network.forward(layers_names_output)
        bounding_boxes = []
        confidences = []
        class_numbers = []

        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

  
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                colour_box_current = colours[class_numbers[i]].tolist()

                cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
                #text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])

                # Putting text with label and confidence on the original image
                #cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 1)


        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

            # concat frame one by one and show result
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
	"""Video streaming route. Put this in the src attribute of an img tag."""
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
	socketio.run(app, host="0.0.0.0", port=5000)