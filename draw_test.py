from graphics import *

def main():
    win = GraphWin("2048", 500, 500)
    
    color = color_rgb(0,0,0)    
    
    win.plot(10,10,color)    
    
    win.getMouse() # pause for click in window
    win.close()
    
main()