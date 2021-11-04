import PySimpleGUI as sg
import time



def Start_test_window(wintitle='TSTS', titles='window title', margins=(200, 100), ):
    layout = [
        [sg.Text(titles[0])],
        [sg.Button('OK')],
        # [sg.Radio],

    ]

    # Create the window
    window = sg.Window(wintitle, layout, margins=margins)

    # Create an event loop
    # keep reading any event from the created window
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break

    window.close()

# https://csveda.com/creating-python-gui-radio-and-checkbox/#:~:text=The%20checkbox%20element%20provided%20by%20pySimplegui%2C%20to%20add,key%2C%20tooltip%2C%20visible%29%20Attributes%20of%20Radio%20and%20Checkbox
def start_toggle_window(wintitle='win', titles=['select additional variables'], margins=(200, 200)):
    layout = [
        [
            [sg.Text(titles[0])],
            [sg.Checkbox("add green travelers", key='GreenTravelers', size=(10, 1), tooltip='Percentage of persons walking, biking, or working from home'),
             sg.Checkbox("savings potential", key='Savings', size=(10, 1),  tooltip='The interaction between solar radiation,  generation potential from RPV,\nand energy cost amounting to the $/day that could be saved\nif all available homes adopted RPV')],
            [sg.Button('EXIT')],
        ]
    ]

    window =sg.Window(wintitle, layout, margins=margins)

    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        print(event)
        for val in v:
            if window.FindElement(val).get==True:
                print('got',  val)

        # time.sleep(10)
