import tkinter as tk
from tkinter import filedialog, Text
import os

#The root frame
root = tk.Tk()
apps = []

if os.path.isfile('save.txt'):
    with open("save.txt", 'r') as f:
        tempApps = f.read()
        tempApps = tempApps.split(',')
        apps = [x for x in tempApps if x.strip()]

def addApp(): 

    for widget in frame.winfo_children():
        widget.destroy()
    
       
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select File",
                                          filetypes = (("executables", "*.exe"), ("all files", "*.*")))
    apps.append(filename)
    print(filename)
    
    for app in apps:
        label = tk.Label(frame, text = app, bg = "grey")
        label.pack()
        
def runApps():
    
    for app in apps:
        os.startfile(app)

#Create a canvas to set the size of a frame
canvas = tk.Canvas(root, height=700, width=700, bg="#263D42")

#Attach the canvas to the root frame                   
canvas.pack()

frame = tk.Frame(root, bg="white")
#Attach the new frame to the root frame and set its size relative to the parent frame 
#Sort of like CSS
frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.1)

#Create a button
openFile = tk.Button(root, text="Open File", 
                     padx=10, pady=5, fg="white", bg="#263D42", command = addApp)
openFile.pack()

runApps = tk.Button(root, text="Run Apps", padx=10, pady=5, fg="white", bg="#263D42", command = runApps)
runApps.pack()

for app in apps:
    label = tk.Label(frame, text = app, bg = "grey")
    label.pack()

root.mainloop()

with open('save.txt', 'w') as f:
    for app in apps:
        f.write(app + ",")