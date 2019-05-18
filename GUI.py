'''
Created on 25 Jan 2019

@author: Christian Ovesen
'''

# GUI related import
from tkinter import * 

# Data storage and database related imports
import sqlite3 as sql
from _sqlite3 import OperationalError

# File interaction
import os

# Logging for error and debugging purpose
import logging



class GUIClass:
# ------------------------------------------------- Constructor -------------------------------------------------
    def __init__(self, window):
        '''    
        Constructor for GUI
        '''
        # Variable for storing selected profile
        self.selectedProfile = StringVar()
        self.selectedProfile.set("None")
        
        # Variable for storing the window
        self.GUIWindow = window
        
        self.profileListCreator()
        
        self.GUICreation()
        self.loginFramePacker()
        self.centerGUIWindow()
        self.canvasIndicatorUpdater()

# ------------------------------------------------- Creators -------------------------------------------------

# Profile list creator
    def profileListCreator(self):
        profilesList = self.getProfiles()
        self.strVarProfilesList = []
        for item in profilesList:
            strVarProfile = StringVar()
            strVarProfile.set(item)
            self.strVarProfilesList.append(strVarProfile)

# Initiate construction of all frames and menus
    def GUICreation(self):
        # Create all frames
        self.loginFrame = Frame(self.GUIWindow, bg="gray13")
        self.areYouSureFrame = Frame(self.GUIWindow, bg="gray13")
        self.profileBackgroundFrame = Frame(self.GUIWindow, bg="gray13")
        self.headsetFrame = Frame(self.profileBackgroundFrame, bg="gray13", highlightthickness=1, highlightcolor="gray85")
        self.neuralNetFrame = Frame(self.profileBackgroundFrame, bg="gray13", highlightthickness=1, highlightcolor="gray85")
        self.dataFrame = Frame(self.profileBackgroundFrame, bg="gray13", highlightthickness=1, highlightcolor="gray85")
        self.profileManagementFrame = Frame(self.profileBackgroundFrame, bg="gray13", highlightthickness=1, highlightcolor="gray85")
        
        # Runs functions for placing widgets in frames
        self.menuCreator()
        self.profileBackgroundFrameCreator()
        self.headsetFrameCreator()
        self.neuralNetFrameCreator()
        self.dataFrameCreator()
        self.profileManagementFrameCreator()
        self.loginFrameCreator()
        self.areYouSureFrameCreator()
        
# Creates top menu
    def menuCreator(self):
        # Top menu
        self.topMenu = Menu(self.GUIWindow)
        GUIWindow.config(menu=self.topMenu)
        # Sub menus
        self.fileMenu = Menu(self.topMenu)
        self.topMenu.add_cascade(label="File", menu=self.fileMenu)
        self.fileMenu.add_command(label="Open profile", command=lambda: self.openProfile("select"))
        self.fileMenu.add_command(label="Delete selected profile", command=self.loginToAreYouSure)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit", command=self.exitProgram)

# Creates background frame for profiles
    def profileBackgroundFrameCreator(self):
        # Creates layout frame with widgets
        self.layoutMenuFrame = Frame(self.profileBackgroundFrame, bg="gray13")
        # Widgets
        self.profileLayoutLabel = Label(self.layoutMenuFrame, text="Layout:", bg="gray13", fg="gray85").grid(row=0, column=0)
        # Drop down menu
        optionList = ("Main", "Training", "Live")
        self.layoutOptionsStrVar = StringVar()
        self.layoutOptionsStrVar.set(optionList[0])
        self.profileMenu = OptionMenu(self.layoutMenuFrame, self.layoutOptionsStrVar, *optionList, command=self.changeProfileLayout)
        self.profileMenu.config(bg="gray13", fg="gray85", activebackground='gray13', activeforeground='gray85')
        self.profileMenu["menu"].config(bg="gray13", fg="gray85")
        self.profileMenu.grid(row=0, column=1)
        
        # Creates connection frame for profile background with widgets
        self.profileConnectionFrame = Frame(self.profileBackgroundFrame, bg="gray13")
        # Widgets
        self.profileConnectionLabel = Label(self.profileConnectionFrame, text="Connection:", bg="gray13", fg="gray85").grid(row=0, column=0)
        # Indicator
        self.profileConnectionCanvas = Canvas(self.profileConnectionFrame, width=15, height=15, highlightthickness=0, bg="gray13")
        self.profileConnectionCanvas.grid(row=0, column=1)
        self.profileConnectionGraphic = self.profileConnectionCanvas.create_oval(0, 0, 15, 15, fill="red2")
        
        # Creates node contact frame for profile background with widgets
        self.profileNodeContactFrame = Frame(self.profileBackgroundFrame, bg="gray13")
        # Widgets
        self.profileNodeContactLabel = Label(self.profileNodeContactFrame, text="Node contact:", bg="gray13", fg="gray85").grid(row=0, column=0)
        # Indicator
        self.profileNodeContactCanvas = Canvas(self.profileNodeContactFrame, width=15, height=15, highlightthickness=0, bg="gray13")
        self.profileNodeContactCanvas.grid(row=0, column=1)
        self.profileNodeContactGraphic = self.profileNodeContactCanvas.create_oval(0, 0, 15, 15, fill="grey13")

# Creates EEG headset frame widgets
    def headsetFrameCreator(self):
        self.testHeadsetLabel = Label(self.headsetFrame, text="TOP LEFT HEADSET FRAME", bg="gray13", fg="gray85").grid(row=1, column=2)
        self.testHeadsetBT = Button(self.headsetFrame, text="Return", command=self.profileToLogin, bg="gray13", fg="gray85").grid(row=2, column=3)
        # Maximize minimize button as canvas
        self.headsetMaximizeMinimizeCanvas = Canvas(self.headsetFrame, width=23, height=23, highlightthickness=1, bg="gray13")
        self.headsetMaximizeMinimizeCanvas.grid(row=0, column=4, sticky=E)
        self.headsetMaximizeMinimizeCanvas.create_image(0, 0, anchor=NE, image=PhotoImage(file="Graphic/Maximize.gif"))
        self.headsetMaximizeMinimizeCanvas.bind("<Button-1>", self.headsetMaximizeFrame)

# Creates neural net frame widgets
    def neuralNetFrameCreator(self):
        self.testNeuralNetLabel = Label(self.neuralNetFrame, text="TOP RIGHT NEURAL NET FRAME", bg="gray13", fg="gray85").grid(row=0, column=2)
        self.testNeuralNetBT = Button(self.neuralNetFrame, text="Return", command=self.profileToLogin, bg="gray13", fg="gray85").grid(row=1, column=3)
        

# Creates data frame widgets
    def dataFrameCreator(self, ):
        self.testDataLabel = Label(self.dataFrame, text="BOTTOM LEFT DATA FRAME", bg="gray13", fg="gray85").grid(row=0, column=2)
        self.testDataBT = Button(self.dataFrame, text="Return", command=self.profileToLogin, bg="gray13", fg="gray85").grid(row=1, column=3)

# Creates profile management frame widgets
    def profileManagementFrameCreator(self, ):
        self.testProfileManagementLabel = Label(self.profileManagementFrame, text="BOTTOM RIGHT PROFILE MANAGEMENT FRAME", bg="gray13", fg="gray85").grid(row=0, column=2)
        self.testProfileMAnagementBT = Button(self.profileManagementFrame, text="Return", command=self.profileToLogin, bg="gray13", fg="gray85").grid(row=1, column=3)

# Creates login frame widgets
    def loginFrameCreator(self, ):
        # Text for profile creation
        self.newProfileInstruction = Label(self.loginFrame, text="Create new profile", bg="gray13", fg="gray85").grid(row=0)
        self.newProfileLabel = Label(self.loginFrame, text="New profile name:", bg="gray13", fg="gray85").grid(row=1, column=0)
        
        # Text for profile selection
        self.selectProfileInstruction = Label(self.loginFrame, text="Choose existing profile", bg="gray13", fg="gray85").grid(row=2)
        self.selectProfileLabel = Label(self.loginFrame, text="Profile name:", bg="gray13", fg="gray85").grid(row=3, column=0)
        
        # Entry field for new profile name
        self.newProfile = Entry(self.loginFrame, bg="gray13", fg="gray85", highlightcolor="gray13")
        self.newProfile.grid(row=1, column=1)
        # Button for creating a new profile with the name given in the entry field
        self.createProfileBT = Button(self.loginFrame, text="Create", command=self.createProfile, bg="gray13", fg="gray85").grid(row=1, column=3)

        # Drop down for all database profile names
        self.profileMenu = OptionMenu(self.loginFrame, self.strVarProfilesList[0], *[x.get() for x in self.strVarProfilesList], command=self.changeSelectedProfile)
        self.profileMenu.config(bg="gray13", fg="gray85", activebackground='gray13', activeforeground='gray85')
        self.profileMenu["menu"].config(bg="gray13", fg="gray85")
        self.profileMenu.grid(row=3, column=1)
        # Button for opening currently selected profile
        self.selectProfileBT = Button(self.loginFrame, text="Select", command=lambda: self.openProfile("select"), bg="gray13", fg="gray85").grid(row=3, column=3)
        self.loginExpand()

# Creates confirmation message for profile deletion
    def areYouSureFrameCreator(self):
        # Labels for deletion confirmation
        self.yesNoLabel = Label(self.areYouSureFrame, text="Are you sure you want to permanently delete this profile?", bg="gray13", fg="gray85").grid(row=0, column=1)
        self.yesNoLabel = Label(self.areYouSureFrame, textvariable=self.selectedProfile, bg="gray13", fg="gray85").grid(row=1, column=1)
        # Yes/No buttons
        self.yesBT = Button(self.areYouSureFrame, text="YES", command=self.deleteProfile, bg="gray13", fg="gray85").grid(row=2, column=0)
        self.noBT = Button(self.areYouSureFrame, text="NO", command=self.areYouSureToLogin, bg="gray13", fg="gray85").grid(row=2, column=2)
        self.areYouSureExpand()
        
# ------------------------------------------------- Updaters -------------------------------------------------

# Updates login frame option menu
    def loginFrameUpdate(self):
        self.profileMenu['menu'].delete(0, 'end')
        profilesList = self.getProfiles()
        self.strVarProfilesList = []
        for item in profilesList:
            strVarProfile = StringVar()
            strVarProfile.set(item)
            self.strVarProfilesList.append(strVarProfile)
        self.profileMenu = OptionMenu(self.loginFrame, self.strVarProfilesList[0], *[x.get() for x in self.strVarProfilesList], command=self.changeSelectedProfile)
        self.profileMenu.config(bg="gray13", fg="gray85", activebackground='gray13', activeforeground='gray85')
        self.profileMenu["menu"].config(bg="gray13", fg="gray85")
        self.profileMenu.grid(row=3, column=1)

# Updates profile specific frames
    def profilesFrameUpdate(self, sourceString):
        if sourceString == "select":
            #TODO: Update from database
            pass
        elif sourceString == "create":
            #TODO: Update for new profile
            pass
        else:
            logging.error("Unknown frame update source")

# Updates canvases
    def canvasIndicatorUpdater(self):
        self.profilesConnectionCanvas()
        self.profilesNodeContactCanvas()

# Updates indicator on canvas in connection frame profiles
    def profilesConnectionCanvas(self):
        # TODO: add proper connection check
        connection = True
        if True == connection:
            self.profileConnectionCanvas.itemconfig(self.profileConnectionGraphic, fill="green2")
        else:
            self.profileConnectionCanvas.itemconfig(self.profileConnectionGraphic, fill="red2")

# Updates indicator on canvas in node change frame profiles
    def profilesNodeContactCanvas(self):
        # TODO: add proper node contact check and flesh out function
        nodeContact = True
        if True == nodeContact:
            self.profileNodeContactCanvas.itemconfig(self.profileNodeContactGraphic, fill="green2")
        else:
            self.profileNodeContactCanvas.itemconfig(self.profileNodeContactGraphic, fill="red2")
        

# ------------------------------------------------- INSERT SECTION -------------------------------------------------



# ------------------------------------------------- Profile functions -------------------------------------------------

# Changes layout for profile related frames
    def changeProfileLayout(self, layoutOption):
        self.mainFrameProfiles(layoutOption)
        pass

# Gets list of profiles in database, creates file if it dosn't exist
    def getProfiles(self):
        profileDatabase = sql.connect("Local/profiles.db")
        c = profileDatabase.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS profiles (
                    name text
                    )""")
        c.execute("SELECT name FROM profiles ORDER BY name ASC")
        profiles = ["None"]
        profile = c.fetchall()
        profile = [x for t in [list(x) for x in profile] for x in t]
        logging.debug(profile)
        profiles = profiles + profile
        profileDatabase.commit()
        profileDatabase.close()
        logging.debug("getProfiles")
        return profiles

# Gets profile by name, returns "None" if it dosn't exist
    def getProfileByName(self, name):
        profileDatabase = sql.connect("Local/profiles.db")
        c = profileDatabase.cursor()
        c.execute("SELECT * FROM profiles WHERE name=:name", {"name": name})
        profileName = c.fetchall()
        profileDatabase.commit()
        profileDatabase.close()
        print("getProfileByName")
        return profileName

# Creates profile
    def createProfile(self):
        logging.debug("createProfile")
        name = self.newProfile.get()
        if name == "":
            logging.error("There is no name entered")
            return
        else:
            pass
        profileDatabase = sql.connect("Local/profiles.db")
        c = profileDatabase.cursor()
        c.execute("SELECT * FROM profiles WHERE name=:name", {"name": name})
        profileName = c.fetchone()
        if profileName == None:
            try:
                c.execute("""CREATE TABLE :name
                    name text
                    """, {"name": name})
            except OperationalError:
                pass
            except:
                logging.error("Unknown error")
                profileDatabase.commit()
                profileDatabase.close()
                return
            c.execute("INSERT INTO profiles(name) VALUES (:name)", {"name": name})
        else:
            logging.error("Profile exists")
            profileDatabase.commit()
            profileDatabase.close()
            return
        profileDatabase.commit()
        profileDatabase.close()
        profile = sql.connect("Local/{}.db".format(name))
        profile.close()
        self.loginFrameUpdate()
        self.openProfile("create")


# Deletes profile
    def deleteProfile(self):
        name = self.selectedProfile.get()
        if self.selectedProfile != "None":
            profileDatabase = sql.connect("Local/profiles.db")
            c = profileDatabase.cursor()
            c.execute("DELETE FROM profiles WHERE name=:name", {"name": name})
            profileDatabase.commit()
            profileDatabase.close()
            fileForDeletion = ("Local/{}.db".format(name))
            if os.path.isfile(fileForDeletion):
                os.remove(fileForDeletion)
            else:
                logging.error("File dosn't exist")
        else:
            logging.error("No profile selected")
        self.loginFrameUpdate()
        self.areYouSureToLogin()

# Changes selected profile when option menu changes selected
    def changeSelectedProfile(self, name):
        self.selectedProfile.set(name)

# Opens profile
    def openProfile(self, sourceString):
        logging.debug("openProfile")
        self.profilesFrameUpdate(sourceString)
        self.loginToProfile()
        # TODO: make function

# ------------------------------------------------- Frame changers -------------------------------------------------

# Changes frames from login to "are you sure"
    def loginToAreYouSure(self):
        if self.selectedProfile.get() == "None":
            logging.error("No profile selected")
        else:
            self.loginFrameUnpacker()
            self.areYouSureFramePacker()
            self.centerGUIWindow()
            self.GUIWindow.state("normal")

# Changes frames from "are you sure" to login
    def areYouSureToLogin(self):
        # Changes shown frame
        self.areYouSureFrameUnpacker()
        self.loginFramePacker()
        self.centerGUIWindow()
        self.GUIWindow.state("normal")

# Changes frames from login to profile specific
    def loginToProfile(self):
        self.loginFrameUnpacker()
        self.profileBackgroundFramePacker()
        self.profilesMainPacker()
        self.centerGUIWindow()
        self.GUIWindow.state("zoomed")
        self.profileBackgroundExpand()

# Changes frames from profile specific to login
    def profileToLogin(self):
        self.profilesUnacker()
        self.profileBackgroundFrameUnpacker()
        self.loginFrameUpdate()
        self.loginFramePacker()
        self.centerGUIWindow()
        self.GUIWindow.state("normal")

# Changes to main frame in profiles
    def mainFrameProfiles(self, layoutOption):
        if "Main" == layoutOption:
            self.profilesUnacker()
            self.profilesMainPacker()
        elif "Training" == layoutOption:
            self.profilesUnacker()
            self.profilesTrainingPacker()
        elif "Live" == layoutOption:
            self.profilesUnacker()
            self.profielsLivePacker()
        else:
            logging.error("No Profile frame layout selected")

# Maximizes headset frame
    def headsetMaximizeFrame(self, imageInfo):
        self.profilesUnacker()
        self.headsetFrameFullPacker()


# ------------------------------------------------- Packers -------------------------------------------------

# Profiles main packers
    def profilesMainPacker(self):
        self.headsetFrameMainPacker()
        self.neuralNetFrameMainPacker()
        self.dataFrameMainPacker()
        self.profileManagementFrameMainPacker()

# Profiles main packers
    def profilesTrainingPacker(self):
        self.neuralNetFrameTrainingPacker()
        self.dataFrameTrainingPacker()

# Profiles live packer
    def profielsLivePacker(self):
        self.dataFrameFullPacker()

# Packs EEG headset frame
    def headsetFrameMainPacker(self):
        self.headsetFrame.grid(row=1, column=0, sticky=NE+SW)

# Packs neural net frame
    def neuralNetFrameMainPacker(self):
        self.neuralNetFrame.grid(row=1, column=1, sticky=NE+SW)

# Packs data frame
    def dataFrameMainPacker(self):
        self.dataFrame.grid(row=2, column=0, sticky=NE+SW)

# Packs profile management frame
    def profileManagementFrameMainPacker(self):
        self.profileManagementFrame.grid(row=2, column=1, sticky=NE+SW)

# Packs neural net frame
    def neuralNetFrameTrainingPacker(self):
        self.neuralNetFrame.grid(row=1, column=0, rowspan=2, sticky=NE+SW)

# Packs data frame
    def dataFrameTrainingPacker(self):
        self.dataFrame.grid(row=1, column=1, rowspan=2, sticky=NE+SW)

# Packs data frame
    def headsetFrameFullPacker(self):
        self.headsetFrame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky=NE+SW)

# Packs data frame
    def neuralNetFrameFullPacker(self):
        self.neuralNetFrame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky=NE+SW)

# Packs data frame
    def dataFrameFullPacker(self):
        self.dataFrame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky=NE+SW)

# Packs data frame
    def profileManagementFrameFullPacker(self):
        self.profileManagementFrame.grid(row=1, column=0, columnspan=2, rowspan=2, sticky=NE+SW)

# Packs login frame
    def loginFramePacker(self):
        self.loginFrame.pack(fill=BOTH, expand=True)

# Packs are you sure frame
    def areYouSureFramePacker(self):
        self.areYouSureFrame.pack(fill=BOTH, expand=True)

# Packs profile background frame
    def profileBackgroundFramePacker(self):
        self.profileBackgroundFrame.pack(fill=BOTH, expand=True)
        self.layoutMenuFrame.grid(row=0, column=0, sticky=W)
        self.profileConnectionFrame.grid(row=3, column=0, sticky=W)
        self.profileNodeContactFrame.grid(row=3, column=1, sticky=E)

# ------------------------------------------------- Unpackers -------------------------------------------------

# Profiles unpackers
    def profilesUnacker(self):
        self.headsetFrameUnpacker()
        self.neuralNetFrameUnpacker()
        self.dataFrameUnpacker()
        self.profileManagementFrameUnpacker()

# Unpacks EEG headset frame
    def headsetFrameUnpacker(self):
        self.headsetFrame.grid_forget()

# Unpacks neural net frame
    def neuralNetFrameUnpacker(self):
        self.neuralNetFrame.grid_forget()

# Unpacks data frame
    def dataFrameUnpacker(self):
        self.dataFrame.grid_forget()
        

# Unpacks profile management frame
    def profileManagementFrameUnpacker(self):
        self.profileManagementFrame.grid_forget()
        

# Unpacks login frame
    def loginFrameUnpacker(self):
        self.loginFrame.pack_forget()

# Unpacks are you sure frame
    def areYouSureFrameUnpacker(self):
        self.areYouSureFrame.pack_forget()

# Unpacks profile background frame
    def profileBackgroundFrameUnpacker(self):
        self.profileBackgroundFrame.pack_forget()

# ------------------------------------------------- Window sizeing -------------------------------------------------

# Centers GUI window
    def centerGUIWindow(self):
        GUIWindow.geometry("")
        GUIWindow.update_idletasks()
        width = GUIWindow.winfo_width()
        height = GUIWindow.winfo_height()
        x = (GUIWindow.winfo_screenwidth() / 2) - (width / 2)
        y = (GUIWindow.winfo_screenheight() / 2) - (height / 2)
        GUIWindow.geometry("%dx%d+%d+%d" % (width, height, x, y))

# Makes columns and rows expand on window resize
    def loginExpand(self):
        # Rows
        self.loginFrame.rowconfigure(0, weight=1)
        self.loginFrame.rowconfigure(1, weight=1)
        self.loginFrame.rowconfigure(2, weight=1)
        self.loginFrame.rowconfigure(3, weight=1)
        # Columns
        self.loginFrame.columnconfigure(0, weight=1)
        self.loginFrame.columnconfigure(1, weight=1)
        self.loginFrame.columnconfigure(2, weight=1)
        self.loginFrame.columnconfigure(3, weight=1)

    def areYouSureExpand(self):
        # Rows
        self.areYouSureFrame.rowconfigure(0, weight=1)
        self.areYouSureFrame.rowconfigure(1, weight=1)
        self.areYouSureFrame.rowconfigure(2, weight=1)
        # Columns
        self.areYouSureFrame.columnconfigure(0, weight=1)
        self.areYouSureFrame.columnconfigure(1, weight=1)
        self.areYouSureFrame.columnconfigure(2, weight=1)

    def profileBackgroundExpand(self):
        # Row
        self.profileBackgroundFrame.rowconfigure(1, weight=1)
        self.profileBackgroundFrame.rowconfigure(2, weight=1)
        # Columns
        self.profileBackgroundFrame.columnconfigure(0, weight=1)
        self.profileBackgroundFrame.columnconfigure(1, weight=1)

    def headsetExpand(self):
        # Rows
        self.headsetFrame.rowconfigure(0, weight=1)
        self.headsetFrame.rowconfigure(1, weight=1)
        self.headsetFrame.rowconfigure(2, weight=1)
        self.headsetFrame.rowconfigure(3, weight=1)
        # Columns
        self.headsetFrame.columnconfigure(0, weight=1)
        self.headsetFrame.columnconfigure(1, weight=1)
        self.headsetFrame.columnconfigure(2, weight=1)
        self.headsetFrame.columnconfigure(3, weight=1)

    def neuralNetExpand(self):
        # Rows
        self.neuralNetFrame.rowconfigure(0, weight=1)
        self.neuralNetFrame.rowconfigure(1, weight=1)
        self.neuralNetFrame.rowconfigure(2, weight=1)
        self.neuralNetFrame.rowconfigure(3, weight=1)
        # Columns
        self.neuralNetFrame.columnconfigure(0, weight=1)
        self.neuralNetFrame.columnconfigure(1, weight=1)
        self.neuralNetFrame.columnconfigure(2, weight=1)
        self.neuralNetFrame.columnconfigure(3, weight=1)

    def dataExpand(self):
        # Rows
        self.dataFrame.rowconfigure(0, weight=1)
        self.dataFrame.rowconfigure(1, weight=1)
        self.dataFrame.rowconfigure(2, weight=1)
        self.dataFrame.rowconfigure(3, weight=1)
        # Columns
        self.dataFrame.columnconfigure(0, weight=1)
        self.dataFrame.columnconfigure(1, weight=1)
        self.dataFrame.columnconfigure(2, weight=1)
        self.dataFrame.columnconfigure(3, weight=1)

    def profileManagementExpand(self):
        # Rows
        self.profileManagementFrame.rowconfigure(0, weight=1)
        self.profileManagementFrame.rowconfigure(1, weight=1)
        self.profileManagementFrame.rowconfigure(2, weight=1)
        self.profileManagementFrame.rowconfigure(3, weight=1)
        # Columns
        self.profileManagementFrame.columnconfigure(0, weight=1)
        self.profileManagementFrame.columnconfigure(1, weight=1)
        self.profileManagementFrame.columnconfigure(2, weight=1)
        self.profileManagementFrame.columnconfigure(3, weight=1)

# ------------------------------------------------- Program closers -------------------------------------------------

# Closes program
    def exitProgram(self):
        logging.debug("exitProgram")
        GUIWindow.destroy()
        exit()

# ------------------------------------------------- INSERT SECTION -------------------------------------------------
GUIWindow = Tk()
GUIobject = GUIClass(GUIWindow)
# Allows for expanding rows and columns when expanding window
# Row
GUIWindow.rowconfigure(0, weight=1)
# Columns
GUIWindow.columnconfigure(0, weight=1)
GUIWindow.mainloop()