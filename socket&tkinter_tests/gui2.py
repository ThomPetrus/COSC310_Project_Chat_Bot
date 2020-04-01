import tkinter as tk

#The root frame
root = tk.Tk()
apps = []

#Create a canvas to set the size of a frame
canvas = tk.Canvas(root, height=700, width=700, bg="#263D42")

#Attach the canvas to the root frame                   
canvas.pack()

frame = tk.Frame(root, bg="white")
#Attach the new frame to the root frame and set its size relative to the parent frame 
#Sort of like CSS
frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.1)

label = tk.Label(frame, text = 'Test')
label.pack()

root.mainloop()