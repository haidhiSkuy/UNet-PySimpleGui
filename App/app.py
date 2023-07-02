import cv2
from model import *
import PySimpleGUI as sg
from PIL import Image, ImageTk

sg.theme('LightGreen2')   

title = [[sg.Text('MRI Brain Tumor Detector', font=("Helvetica", 12, "bold"), pad=(50,30))], 
         [sg.Image('assets/brain2_resized.png', pad=(90,(0,30)))],
         [sg.HorizontalSeparator()],
         [sg.Text('Select Image',pad=(100,10))],
         [sg.FileBrowse(target='Browse', enable_events=True,key='Browse',pad=(115,0))],
         [sg.Image('assets/upload_image.png',filename="", key="-IMAGE-",pad=(60,20))],
         [sg.Button("Predict",key='-UNET-',enable_events=True,pad=(115,(0,20)))],
         [sg.HorizontalSeparator()],
         [sg.Text('Created by : ', font=("Helvetica",11,"bold"))],
         [sg.Text('Haidhi Angkawijana Tedja')]]

image_layout = [[sg.Image('assets/segmentation_result.png',filename="", key="-PREDICTION-",pad=((60,10)))], 
                [sg.Image('assets/result_merged.png',filename="", key="-MERGED-", pad=((60,10),(30,0)))],
                [sg.FileSaveAs(key='-SAVE-',enable_events=True, pad=((130,0),(30,0)))]]

layout = [[sg.Column(title),sg.VSeparator(),sg.Column(image_layout)]]

window = sg.Window('Tumor Detector',layout, size=(680, 600), grab_anywhere=True,icon=r'assets/brain.ico')

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': 
        break
    
    if event == 'Browse':
        image = cv2.imread(values['Browse']) 
        window["-PREDICTION-"].update(filename='assets/segmentation_result.png')
        window["-MERGED-"].update(filename='assets/result_merged.png')
        window["-IMAGE-"].update(data=cv2.imencode(".png", cv2.resize(image, (150,150)))[1].tobytes())

        # Used for predicting
        image = cv2.resize(image, (256,256))

    #PREDICTING
    if event == '-UNET-':
        try:
            raw_result, segmentation_result = predict(image)
            merged_result = merged(image,raw_result)

            window["-PREDICTION-"].update(data=segmentation_result)
            window["-MERGED-"].update(data=merged_result[0])
           
        except: 
            sg.popup_ok("Please select your image first",title='Warning')

    #Save file 
    if event == '-SAVE-':
        save_path = str(values['-SAVE-'])
        try:
            cv2.imwrite(save_path,merged_result[1])
        except: 
            sg.popup_ok("Make Prediction for Your Brain Image First",title='Warning',)

window.close()