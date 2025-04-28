
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from torchvision.models import detection
import sqlite3
import torch
from torchvision import models
from flask import Flask, render_template, request, redirect, Response
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


app = Flask(__name__)


model = torch.hub.load("ultralytics/yolov5", "custom", path = "best.pt", force_reload=True)

model.eval()
model.conf = 0.5  
model.iou = 0.45  

def get_email_from_username(username):
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("SELECT email FROM `info` WHERE user=?", (username,))
    result = cur.fetchone()
    con.close()
    if result:
        return result[0]  
    else:
        return None

from io import BytesIO

def gen():
    """
    The function takes in a video stream from the webcam, runs it through the model, and returns the
    output of the model as a video stream
    """
    cap=cv2.VideoCapture(0)
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=415)
            results.print()  
            img = np.squeeze(results.render()) 
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        else:
            break
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    """
    It returns a response object that contains a generator function that yields a sequence of images
    :return: A response object with the gen() function as the body.
    """
    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    global user
    """
    The function takes in an image, runs it through the model, saves the output image to a
    static folder, and sends an email with the predicted classes and image.
    :return: The image is being returned.
    """
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=415)
        classes = results.names

        for img in results.render():
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")

        email = get_email_from_username(user)
        sender_email = "myprojectstp@gmail.com"  
        receiver_email = email
        password = "paxgxdrhifmqcrzn"  

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = "YoloV5 Predictions"

        # Add body to email
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        body = 'Wild Animal found at ' + str(current_time)
        message.attach(MIMEText(body, "plain"))

        # Attach image to email
        img_data = open("static/image0.jpg", "rb").read()
        image = MIMEImage(img_data, name="image0.jpg")
        message.attach(image)

        # Connect to the SMTP server
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())

        return redirect("static/image0.jpg")
    return render_template("index.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "myprojectstp@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("myprojectstp@gmail.com", "paxgxdrhifmqcrzn")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")



@app.route("/signin")
def signin():
    global user

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    user = mail1
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/notebook")
def notebook():
    return render_template("WildAnimalDetection.html")

if __name__ == "__main__":
    app.run(port=5000)  
