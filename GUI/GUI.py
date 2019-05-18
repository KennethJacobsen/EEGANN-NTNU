'''
Created on 28 Mar 2019

@author: Christian Ovesen
'''
# -------------- Imports --------------
# GUI related import
from tkinter import *
from tkinter import ttk
# Logging for error and debugging purpose
import logging
# Multithreading
import threading
from threading import Lock
import queue
import multiprocessing
# Sleep
import time
# Data management
import numpy as np
#from .. import Logic
import sys
# sys.path.append("../")
# from Logic import *
# TODO: Figure out how to do commands
# File format eeg data
# [36.0, 0.0, 0.0, 4112.949, 4109.487, 4118.846, 4111.154, 4111.538, 4106.795, 4103.462, 4109.744, 4110.513, 4114.615,
# 4168.974, 4892.308, 4109.744, 4118.59, 0.0, 0.0]


# -------------- Creators --------------



# -------------- Layouts --------------



# -------------- Frame change --------------

# Open window showing commands to record
def commandsWindow(window, lock, q):
    commandWindow = GUIwindow(window)
    commandFrame = GUIframe(commandsWindow)
    newLabel(commandFrame, labelText="Think of the commands as they show up in this window", span=[3, 1])
    commandFrame.framePacker()
    
    commandWindow.windowedNoramlize()
    commandWindow.centerWindow()
    commandWindow.minSize()
    
    gatheringData = True
    messageRecived = False
    while gatheringData:
        if q.qsize() > 0:
            messageRecived = True
            lock.acquire()
            returnMessage = q.get()
            lock.release()
        
        if returnMessage == "EndCmds":
            gatheringData = False
        
        if messageRecived:
            commandFrame.frameUnpacker()
            commandFrame.GUIframe.destroy()
            commandFrame = GUIframe(commandWindow)
            newLabel(commandFrame, labelText=returnMessage, span=[3, 1])
            pass
        
        messageRecived = False
    
    commandWindow.closeWindow()


# -------------- Widgets --------------

# Create label
def newLabel(frame, labelText="", textVariable="", position=None, span=None, stickySide="nsew"):
    if position is None:
        position = [0, 0]
    if span is None:
        span = [1, 1]
    if textVariable == "":
        newLabel = ttk.Label(frame, text=labelText)
    else:
        newLabel = ttk.Label(frame, textvariable=textVariable)
    if stickySide != "":
        newLabel.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1], sticky=stickySide)
    else:
        newLabel.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1])


# Create button
def newButton(frame, internalCommand, buttonText="", position=None, span=None, stickySide=""):
    if position is None:
        position = [0, 0]
    if span is None:
        span = [1, 1]
    newBT = ttk.Button(frame, text=buttonText, command=internalCommand)
    if stickySide != "":
        newBT.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1], sticky=stickySide)
    else:
        newBT.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1])


# Create entry field
def newEntry(frame, position=None, span=None, stickySide=""):
    if position is None:
        position = [0, 0]
    if span is None:
        span = [1, 1]
    newEntry = ttk.Entry(frame)
    if stickySide != "":
        newEntry.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1], sticky=stickySide)
    else:
        newEntry.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1])


# Create drop down menu
# TODO: postcommand to run a function updateing values
def newCombobox(frame, entryList=None, position=None, span=None, stickySide=""):
    if entryList is None:
        entryList = [""]
    if position is None:
        position = [0, 0]
    if span is None:
        span = [1, 1]
    comboList = ["None"]
    for x in entryList:
        comboList.append(x)
    cb = ttk.Combobox(frame, values=comboList)
    cb.set(comboList[0])
    if stickySide != "":
        cb.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1], sticky=stickySide)
    else:
        cb.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1])

# -------------- Widgets --------------

# Differentiate what combobox was used
#def comboboxDistinction(comboTagg, value):
#    if comboTagg == "ProfileSelection":
#        changeSelectedProfile(value)
#    elif comboTagg == "ProfileLayout":
#        changeProfileLayout(value)
#    else:
#        logging.error("Not a valid Combobox")


def findInGrid(frame, position):
    for children in frame.getChildren():
        info = children.grid_info()
        # Note that position numbers are stored as string                                                                         
        if info['row'] == str(position[0]) and info['column'] == str(position[1]):
            return children
    return None

# -------------- Style --------------

def set_app_style():
    style = ttk.Style()
    style.theme_create( "EEGANN_app", parent="alt", settings={
        ".":             {"configure": {"background"      : "gray13",
                                        "foreground"      : "gray85",
                                        "relief"          : "flat",
                                        "highlightcolor"  : "gray50"}},
        "TFrame":        {"configure": { "background"     : "gray13"}},

        "Toplevel":      {"configure": { "background"     : "gray13"}},

        "TLabel":        {"configure": {"foreground"      : "gray85",
                                        "padding"         : 1,
                                        "font"            : ("Calibri", 12)}},

        "TNotebook":     {"configure": {"padding"         : 5}},
        "TNotebook.Tab": {"configure": {"padding"         : [25, 5], 
                                        "foreground"      : "gray85"},
                            "map"      : {"background"    : [("selected", "gray13")],
                                        "expand"          : [("selected", [1, 1, 1, 0])]}},

        "TCombobox":     {"configure": {"selectbackground": "gray13",
                                        "fieldbackground" : "gray70",
                                        "background"      : "gray13",
                                        "foreground"      : "gray13"}},

        "TButton":       {"configure": {"font"            :("Calibri", 13, 'bold'),
                                        "background"      : "gray18",
                                        "foreground"      : "gray85"},
                                        "borderwidth"     :10,
                                        "bordercolor"     : "gray85",
                                        
                            "map"      : {"background"    : [("active", "gray85")],
                                        "foreground"      : [("active", "gray13")]}},
            
        "TEntry":        {"configure": {"background"      : "gray13",
                                        "foreground"      : "gray13",
                                        "highlightcolor"  : "gray13"}},
        "Horizontal.TProgressbar":{"configure": {"background": "gray13"}}
    })
    style.theme_use("EEGANN_app")
    

# -------------- Canvas functions --------------

# Resizable canvas
class resCanvas(Canvas):
    def __init__(self, parent,  **kwargs):
        Canvas.__init__(self, parent, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all", 0, 0, wscale, hscale)


# -------------- GUIframe Class --------------

class GUIframe:
    '''
    GUIframe configures frames for the GUI with standard settings applied
    '''
    def __init__(self, window):
        '''
        Constructor
        '''
        # Creates frame
        self.window = window
        self.GUIframe = ttk.Frame(window.GUIWindow)

    # -------------- GUIframe Layout --------------
    # Change layout and weights of frame
    def weightChange(self, size):
        x = 0
        # Row
        for index in size[0]:
            self.GUIframe.rowconfigure(x, weight=size[0][x])
            x += 1
        x=0
        # Columns
        for index in size[1]:
            self.GUIframe.columnconfigure(x, weight=size[1][x])
            x += 1

    # -------------- Children --------------
    def getChildren(self):
        frameChildren = self.GUIframe.children.values()
        return frameChildren

    # -------------- Change window sizing --------------
    def normalizeWindow(self):
        if self.windowClass != None:
            self.windowClass.windowedNoramlize()

    def fullscreenWindow(self):
        if self.windowClass != None:
            self.windowClass.windowedFullscreen()
    
    def setMinSize(self):
        if self.windowClass != None:
            self.windowClass.minSize()

    # -------------- GUIframe Packing --------------
    # Packers
    def framePacker(self, position=None, span=None):
        if position is None:
            position = [0, 0]
        if span is None:
            span = [1, 1]
        self.GUIframe.grid(row=position[0], column=position[1], rowspan=span[0], columnspan=span[1], sticky="nsew")

    # Unpacker
    def frameUnpacker(self):
        self.GUIframe.grid_forget()

# -------------- GUIframe Class --------------

class GUIwindow():
    '''
     GUIwindow configures windows for the GUI with standard settings applied
    '''
    
    def __init__(self, root, name, current="", layoutWeights=[[1],[1]]):
        '''
        Constructor
        '''
        self.root = root
        self.window = Toplevel(root)
        self.window.title(name)
        if current == "topMenu":
            self.menuCreator()
        self.weightChange(layoutWeights)
        self.window.update()
        
        self.centerWindow()

    # -------------- GUIwindow sizing --------------
    
    # Set minimum size of window
    def minSize(self):
        self.window.update()
        self.window.minsize(self.window.winfo_width(), self.window.winfo_height())

    # Centers GUI window
    def centerWindow(self):
        self.window.geometry("")
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() / 2) - (width / 2)
        y = (self.window.winfo_screenheight() / 2) - (height / 2)
        self.window.geometry("%dx%d+%d+%d" % (width, height, x, y))

    # Windowed fullscreen
    def windowedFullscreen(self):
        self.window.state("zoomed")

    # Windowed normalize
    def windowedNoramlize(self):
        self.window.state("normal")
        self.centerWindow()

    # Change window layout
    def weightChange(self, size):
        x = 0
        # Row
        for index in size[0]:
            self.window.rowconfigure(x, weight=size[0][x])
            x += 1
        x=0
        # Columns
        for index in size[1]:
            self.window.columnconfigure(x, weight=size[1][x])
            x += 1

    # -------------- GUIwindow closers --------------
    
    # Closes program
    def closeWindow(self):
        logging.debug("Closed window")
        self.window.destroy()

    def killRootActivation(self):
        self.window.protocol("WM_DELETE_WINDOW", self.root.destroy)

# -------------- Constructor --------------

# Communicates with logic through queues
class comClass():
    """
    Class doc
    """
    def __init__(self, mainWindow, lock, logicComQ, returnQ):
        self.mainWindow = mainWindow
        self.lock = lock
        self.logicComQ = logicComQ
        self.returnQ = returnQ

    def logicCom(self, commandLibrary):
        # Library format: {"Command": "Kill", "List": [x, y, z]}
        # Delete profile format: {"Command": "DeleteProfiel", "Profile": "*Profile Name*"}
        # Train examples: {'Command': 'Train', 'Outputs': ['Up', 'Down'], 'Variables': {'Type': 'LSTM', 'Layers': 3, 'Neurons': [180, 150, 120, 90]}}
        self.lock.acquire()
        self.logicComQ.put(commandLibrary)
        self.lock.release()
        self.returnQ()

    def returnQ(self):
        returnWait = True
        while returnWait:
            if self.returnQ.qsize() > 0:
                self.lock.acquire()
                returnMessage = self.returnQ.get()
                self.lock.release()
                
                if returnMessage == "Successful":
                    returnWait = False
                    break
                elif returnMessage == "InitCmds":
                    commandsWindow(self.mainWindow, self.lock, self.returnQ)
                    returnWait = False
                    break
                # TODO: Add more depending on needs. Example: Profile commands
                else:
                    errorWindow = GUIwindow(self.root, "Error")
                    errorFrame = GUIframe(errorWindow)
                    errorFrame.framePacker()
                    newLabel(errorFrame, labelText="Error:", span=[1, 3])
                    newLabel(errorFrame, labelText=returnMessage, position=[1, 0], span=[1, 3])
                    newButton(errorFrame, internalCommand=lambda: errorWindow.closeWindow(), buttonText="OK", position=[2, 2])


# -------------- Constructor --------------

# Manages windows and frames
class windowFrameManager():
    """
    Class doc
    """
    def __init__(self, root, lock, eegQ, devQ, logicComQ, returnQ):
        self.lock = lock
        self.eegQ = eegQ
        self.devQ = devQ
        self.mainWindow = GUIwindow(root, "EEG-ANN app", current="topMenu")
        self.mainWindow.killRootActivation()
        self.menuCreator(root)
        self.comClass = comClass(self.mainWindow, lock, logicComQ, returnQ)
    
    # Creates top menu on main window
    def menuCreator(self, window):
        # Top menu
        topMenu = Menu(self.window)
        window.config(menu=self.topMenu)
        # Sub menus
        fileMenu = Menu(self.topMenu)
        topMenu.add_cascade(label="File", menu=self.fileMenu)
        # fileMenu.add_command(label="Open profile", command=lambda: openProfile("select"))
        fileMenu.add_command(label="Delete selected profile", command=lambda: self.delConfFrameCreator(window))
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=lambda: self.mainWindow())

    
    def loginScreen(self):
        # Create login frame
        self.loginFrame = GUIframe(self.mainWindow)
        # Add all components to login frame
        self.loginFrameCreator(self.loginFrame)
        # Set minimum size of window
        self.mainWindow.windowedNoramlize()
        self.mainWindow.minSize()
    
    def profileScreen(self):
        # Creating profile screen related frames
        self.profileLayoutFrame = GUIframe(self.mainWindow)
        self.layoutMenuFrame = GUIframe(self.profileLayoutFrame)
        self.profileConnectionFrame = GUIframe(self.profileLayoutFrame)
        self.headsetFrame = GUIframe(self.profileLayoutFrame)
        self.neuralNetFrame = GUIframe(self.profileLayoutFrame)
        self.dataFrame = GUIframe(self.profileLayoutFrame)
        self.profileManagementFrame = GUIframe(self.profileLayoutFrame)
        # Frames for connection and graph graphics
        self.conFrame = GUIframe(self.dataFrame)
        self.graphFrame = GUIframe(self.dataFrame)
        # Add all components to profile background frame
        self.profileBackgroundFrameCreator(self.profileLayoutFrame, self.layoutMenuFrame,
                                          self.profileConnectionFrame, self.headsetFrame, self.neuralNetFrame,
                                          self.dataFrame, self.profileManagementFrame)


        
        # Creates login frame widgets
    def loginFrameCreator(self, frame):
        # Main profile setup
        """
        # Text for profile creation
        newLabel(loginFrame.GUIframe, labelText="Create new profile", span=[1, 4], stickySide='')
        newLabel(loginFrame.GUIframe, labelText="New profile name:", position=[1, 0])
            
        # Text for profile selection
        newLabel(loginFrame.GUIframe, labelText="Choose existing profile", position=[2, 0], span=[1, 4], stickySide='')
        newLabel(loginFrame.GUIframe, labelText="Profile name:", position=[3, 0])
            
        # Entry field for new profile name
        newEntry(loginFrame.GUIframe, position=[1, 1])
        
        # Button for creating a new profile with the name given in the entry field
        newButton(loginFrame.GUIframe, buttonText="Create", internalCommand=lambda: createProfile(loginFrame,
                                                                                         profileLayoutFrame, dataFrame,
                                                                                         conFrame, graphFrame,
                                                                                         graphConnection), position=[1, 3])
        
        # Drop down for all database profile names
        # profileMenu = OptionMenu(loginFrame, profileListCreator(profilesList), command=lambda: changeSelectedProfile(),
        # position=[3,1])
        newCombobox(loginFrame.GUIframe, entryList=["TODO:", "get lits", "of profiles"], position=[3, 1])
        # Button for opening currently selected profile
        newButton(loginFrame.GUIframe, buttonText="Select", internalCommand=lambda: openProfile("select"), position=[3, 3])
        """
        # Temp profile setup
        newButton(frame, internalCommand=lambda: self.loginToProfile(), buttonText="To profile screen")
        
        # Pack frame and set weights
        frame.weightChange([[1, 1, 1, 1], [1, 1, 1, 1]])
        frame.framePacker()
    
    # Creates background frame for profiles
    def profileBackgroundFrameCreator(self, frame, topBarFrame, bottomBarFrame, headsetFrame, neuralNetFrame, dataFrame, profileManagementFrame):
        
        # Frame managing
        # Profile frame layout 
        frame.weightChange([[0, 1, 1, 0], [1, 1]])
        
        # Top bar frame layout
        topBarFrame.weightChange([[1], [0, 0, 1]])
        topBarFrame.framePacker(span=[1, 2])
        
        # Bottom bar frame layout
        bottomBarFrame.weightChange([[1], [0, 0, 1, 0, 0]])
        bottomBarFrame.framePacker(position=[3, 0], span=[1, 2])
        
        self.mainLayout(headsetFrame, neuralNetFrame, dataFrame, profileManagementFrame)

        # Top bar widgets
        newLabel(topBarFrame.GUIframe, labelText="Layout:", stickySide="w")
        
        # Drop down menu
        optionList = ["Main", "Training", "Live"]
        profileCB = newCombobox(topBarFrame.GUIframe, entryList=optionList, position=[0,1], stickySide="w")
        profileCB.bind('<<ComboboxSelected>>', self.profileLayoutChange(profileCB.get()), headsetFrame, neuralNetFrame, dataFrame, profileManagementFrame)
        
        # Bottom bar widgets
        newLabel(bottomBarFrame.GUIframe, labelText="Connection:", stickySide="w")
        newLabel(bottomBarFrame.GUIframe, labelText="Node contact:", position=[0, 3], stickySide="e")
        
        # Headset frame widgets
        newLabel(headsetFrame.GUIframe, labelText="Connect to cortex headset:", span=[1, 2])
        newButton(headsetFrame, internalCommand=lambda: self.cortexStart(), buttonText="Connect", position=[0, 2])
        
        # Neural net frame widgets
        
        # Data frame widgets
        
        # Profile management frame widgets
        newLabel(profileManagementFrame.GUIframe, labelText="", span=[1, 3])
        newButton(profileManagementFrame.GUIframe, internalCommand=lambda: self.profileToLogin(), buttonText="Return",
                  position=[1, 3])
        
        # Place frame in window
        frame.framePacker()
        # Set new minimum size of window
        frame.normalizeWindow()
        frame.setMinSize()
        # Set window to fullscreen
        frame.fullscreenWindow()
        
        """
        # Indicator
        self.connectionCanvas = Canvas(self.profileConnectionFrame, width=15, height=15, highlightthickness=0, bg="gray13")
        self.connectionCanvas.grid(row=0, column=1)
        self.connectionGraphic = self.profileConnectionCanvas.create_oval(0, 0, 15, 15, fill="red2")

        # Indicator
        self.contactOverviewCanvas = Canvas(self.profileNodeContactFrame, width=15, height=15, highlightthickness=0, bg="gray13")
        self.contactOverviewCanvas.grid(row=0, column=4)
        self.contactOverviewCanvas = self.profileNodeContactCanvas.create_oval(0, 0, 15, 15, fill="grey13")
        """
    
    # -------------- Profile layouts --------------

    
    def mainLayout(self, headsetFrame, neuralNetFrame, dataFrame, profileManagementFrame):
        # Headset frame layout
        headsetFrame.weightChange([[0, 1], [1, 1, 1]])
        headsetFrame.framePacker(position=[1, 0])
        
        # Neural net frame layout
        neuralNetFrame.weightChange([[0, 1], [1, 1, 1]])
        neuralNetFrame.framePacker(position=[1, 1])
    
        # Data frame layout
        dataFrame.weightChange([[0, 1], [1, 2]])
        dataFrame.framePacker(position=[2, 0])
        
        # Profile management frame layout
        profileManagementFrame.weightChange([[0, 1], [1, 1, 1]])
        profileManagementFrame.framePacker(position=[2, 1])
    
    def trainLayout(self, neuralNetFrame, dataFrame):
        # Neural net frame layout
        neuralNetFrame.weightChange([[0, 1], [1, 1, 1]])
        neuralNetFrame.framePacker(position=[1, 1], span=[2, 1])
        
        # Data frame layout
        dataFrame.weightChange([[0, 1], [1, 2]])
        dataFrame.framePacker(position=[1, 0], span=[2, 1])
    
    def liveLayout(self, dataFrame):
        # Data frame layout
        dataFrame.weightChange([[0, 1], [1, 2]])
        dataFrame.framePacker(position=[1, 0], span=[2, 2])
    
    # -------------- Frame changers --------------
    
        # Change from login frame to profile frame
    def loginToProfile(self):
        # Removes login frame from window and destroys it
        self.loginFrame.frameUnpacker()
        self.loginFrame.GUIframe.destroy()
        
        # Creates and puts profile screen into window
        self.profileScreen()
    
    # Change from profile frame to login frame
    def profileToLogin(self):
        # Removes profile layout frame from window and destroys it
        self.profileLayoutFrame.frameUnpacker()
        self.profileLayoutFrame.GUIframe.destroy()
        
        # Creates and puts login screen into window
        self.loginScreen()
    
    def profileLayoutChange(self, layout, headsetFrame, neuralNetFrame, dataFrame, profileManagementFrame):
        # Unpack frames in profile
        headsetFrame.frameUnpacker()
        neuralNetFrame.frameUnpacker()
        dataFrame.frameUnpacker()
        profileManagementFrame.frameUnpacker()
        # Destroy frames
        #headsetFrame.GUIframe.destroy()
        #neuralNetFrame.GUIframe.destroy()
        #dataFrame.GUIframe.destroy()
        #profileManagementFrame.GUIframe.destroy()
        
        self.profileLayout = layout
        
        if layout == "Main":
            headsetFrame.framePacker(position=[1, 0])
            neuralNetFrame.framePacker(position=[1, 1])
            dataFrame.framePacker(position=[2, 0])
            profileManagementFrame.framePacker(position=[2, 1])
        elif layout == "Training":
            neuralNetFrame.framePacker(position=[1, 1], span=[2, 1])
            dataFrame.framePacker(position=[1, 0], span=[2, 1])
        elif layout == "Live":
            dataFrame.framePacker(position=[1, 0], span=[2, 2])
    
    # -------------- Extra windows --------------
    
    # Creates confirmation message for profile deletion
    def delConfFrameCreator(self):
        #
        profileName = "TODO: get Combobox profile selection"
        if profileName != "None":
            delConf = GUIwindow(self.mainWindow, "Deletion confirmation")
            delConfFrame = GUIframe(delConf.window)
        else:
            logging.error("No profile slected")
            return
        
        # Create library with message to logic
        commandLibrary = {"Command": "DeleteProfiel", "Profile": profileName}
        # Labels for deletion confirmation
        newLabel(delConfFrame.GUIframe, labelText="Are you sure you want to permanently delete this profile?", position=[0, 1], span=[1, 3])
        newLabel(delConfFrame.GUIframe, textVariable="TODO: add currently selected profile", position=[1, 1])
        # Yes/No buttons
        newButton(delConfFrame.GUIframe, buttonText="YES", internalCommand=lambda: lambda: self.comClass.logicCom(commandLibrary), position=[2, 0], span=[1, 2])
        newButton(delConfFrame.GUIframe, buttonText="NO", internalCommand=lambda: delConf.closeWindow(), position=[2, 2], span=[1, 2])
        
        # Pack frame and set weights
        delConfFrame.weightChange([[1, 1, 1], [1, 1, 1, 1]])
        delConfFrame.framePacker()
    
    # -------------- Commands --------------
    
    def cortexStart(self):
        # Create library with message to start cortex connection for "Logic.py" module in headset frame
        commandLibrary = {"Command": "StartCortex"}
        # Sends command to Logic.py
        self.comClass.logicCom(commandLibrary)
        
        # Opening thread for connection and graph graphics
        self.dataGraphConnection = dataThread(lock=self.lock, eegQ=self.eegQ, devQ=self.devQ)
        self.dataGraphConnection.run(self.dataFrame.GUIframe, self.conFrame.GUIframe, self.graphFrame.GUIframe) 
        
        self.conFrame.weightChange([[1], [1]])
        self.conFrame.framePacker(position=[1, 1])
        self.graphFrame.weightChange([[1], [1]])
        self.graphFrame.framePacker(position=[1, 0])


# -------------- Threading and multiprocessing --------------

# TODO: Possibly move/implement to GUI?
# Resizable GUI for graph and connection.
# Class used to create second thread for GUI
class guiThread(threading.Thread):
    """
    Class doc
    """
    def __init__(self, lock, eegQ, devQ, logicComQ, returnQ):
        threading.Thread.__init__(self)
        root = self.start()
        self.windowFrameManager = windowFrameManager(root, lock, eegQ, devQ, logicComQ, returnQ)
        self.root.mainloop()

    def run(self):
        # Variable for storing selected profile
        root = Tk()
        root.withdraw()
        root.protocol("WM_DELETE_WINDOW", lambda: self.root.destroy())
        set_app_style()
        return root

# Resizable GUI for graph and connection.
# Class used to create second process for GUI
class guiProcess(multiprocessing.Process):
    """
    Class doc
    """
    def __init__(self, lock, eegQ, devQ, logicComQ, returnQ):
        threading.Thread.__init__(self)
        root = self.start()
        self.windowFrameManager = windowFrameManager(root, lock, eegQ, devQ, logicComQ, returnQ)
        self.root.mainloop()

    def run(self):
        # Variable for storing selected profile
        root = Tk()
        root.withdraw()
        root.protocol("WM_DELETE_WINDOW", lambda: self.root.destroy())
        set_app_style()
        return root

# -------------- Kenneth GUI code --------------

class dataThread(threading.Thread):
    """
    Class doc
    """
    def __init__(self, lock, eegQ, devQ):
        threading.Thread.__init__(self)
        self.lock = lock
        self.eegDQ = eegQ
        self.devQ = devQ

    def run(self, frame, conFrame, graphFrame):
        self.graphCanvas = resCanvas(graphFrame, width=800, height=800, bg="gray13",
                                     highlightthickness=0)
        self.conCanvas = resCanvas(conFrame, width=800, height=800, bg="gray13",
                                   highlightthickness=0)
        self.graphCanvas.grid(row=1, column=1, sticky='nesw')
        self.conCanvas.grid(row=1, column=0, sticky='nesw')
        newLabel(frame, labelText="Sensor Graph", position=[0, 0])
        newLabel(frame, labelText="Sensor Contact", position=[0, 1])

        colorsensors = ['black', 'red', 'red', 'yellow', 'green']
        colorgraphs = ['cyan', 'green', 'blue', 'LightBlue4', 'gold2', 'tan1', 'red3', 'maroon1', 'purple1', 'coral1',
                       'orange', 'saddle brown', 'goldenrod', 'gold']
        startcord = [[190, 70], [110, 145], [200, 125], [180, 180], [65, 230], [110, 360], [200, 430], [310, 430],
                     [400, 360], [445, 230], [330, 180], [310, 125], [400, 145], [320, 70]]

        while True:
            x2 = 0
            self.lock.acquire()
            array = self.eegDQ.queue[0].copy()
            self.lock.release()

            self.lock.acquire()
            dq = self.devQ.queue[0].copy()
            self.lock.release()

            if len(array[0]) > 0:
                for i in range(14):
                    array[i] = array[i][-512:]# * 0.125
                    array[i] = [x * 0.125 for x in array[i]]
                    array[i] = np.subtract(array[i], min(array[i]))
                    array[i] = np.subtract(array[i], max(array[i]) / 2)

            self.graphCanvas.delete("all")
            dWidth1, dHeight1 = 500, 500
            xOffset1, yOffset1 = 20, 20

            self.conCanvas.delete("all")
            dWidth2, dHeight2 = 500, 500
            xOffset2, yOffset2 = 50, 50

            sh = float(self.graphCanvas.winfo_height())/(dHeight1+yOffset1)
            sw = float(self.graphCanvas.winfo_width())/(dWidth1+xOffset1)
            for i in range(len(array[0]) - 1):
                x1 = x2
                x2 = i + 1
                for i2 in range(14):
                    self.graphCanvas.create_line((x1+xOffset1)*sw*0.85,
                                                 (array[i2][i] + yOffset1 + ((i2 + 1)*35))*sh*0.9,
                                                 (x2+xOffset1)*sw*0.85,
                                                 (array[i2][i + 1] + yOffset1 + ((i2 + 1)*35))*sh*0.9,
                                                 fill=colorgraphs[i2])
            self.graphCanvas.addtag_all("all")
            self.graphCanvas.update()

            sh = float(self.conCanvas.winfo_height())/(dHeight2+yOffset2)
            sw = float(self.conCanvas.winfo_width())/(dWidth2+xOffset2)

            self.conCanvas.create_oval(dWidth2*sw, dHeight2*sh, xOffset2*sw, yOffset2*sh, fill='royal blue')
            for i in range(14):
                color = colorsensors[int(dq[2][i])]
                self.conCanvas.create_oval((startcord[i][0] + xOffset2-50)*sw, startcord[i][1]*sh,
                                           (startcord[i][0] + 40)*sw, (startcord[i][1] + 40)*sh, fill=color,
                                           outline=color)

            self.graphCanvas.addtag_all("all")
            self.graphCanvas.update()
            self.conCanvas.addtag_all("all")
            self.conCanvas.update()
            time.sleep(0.1)