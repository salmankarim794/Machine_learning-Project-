import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as mlt
from tkinter import messagebox

global var_result
import sys

matches=pd.read_csv('C:\\Users\\salman\\Desktop\\MATCHES_DATA.csv')
matches.info()

#finding all Not available values in winner column, so that we update this as draw
matches[pd.isnull(matches['winner'])]
matches['winner'].fillna('Draw', inplace=True)


matches.replace(['Pakistan','England','India','Australia','Zimbabwe',
                 'Bangladesh','Sri Lanka','Scotland','Afghanistan',
                 'Hong Kong','West Indies','South Africa','United Arab Emirates','Ireland','New Zealand','Netherlands']
                ,['Pakistan','England','India','Australia','Zimbabwe','Bangladesh','Sri Lanka','Scotland','Afghanistan','Hong Kong','West Indies','South Africa','United Arab Emirates','Ireland','New Zealand','Netherlands'],inplace=True)

encode = {'team1': {'Pakistan':1,'England':2,'India':3,'Australia':4,'Zimbabwe':5,'Bangladesh':6,'Sri Lanka':7,'Scotland':8,'Afghanistan':9,'Hong Kong':10,'West Indies':11,'South Africa':12,'United Arab Emirates':13,'Ireland':14,'New Zealand':15,'Netherlands':16},
          'team2': {'Pakistan':1,'England':2,'India':3,'Australia':4,'Zimbabwe':5,'Bangladesh':6,'Sri Lanka':7,'Scotland':8,'Afghanistan':9,'Hong Kong':10,'West Indies':11,'South Africa':12,'United Arab Emirates':13,'Ireland':14,'New Zealand':15,'Netherlands':16},
          'toss_winner': {'Pakistan':1,'England':2,'India':3,'Australia':4,'Zimbabwe':5,'Bangladesh':6,'Sri Lanka':7,'Scotland':8,'Afghanistan':9,'Hong Kong':10,'West Indies':11,'South Africa':12,'United Arab Emirates':13,'Ireland':14,'New Zealand':15,'Netherlands':16},
          'winner': {'Pakistan':1,'England':2,'India':3,'Australia':4,'Zimbabwe':5,'Bangladesh':6,'Sri Lanka':7,'Scotland':8,'Afghanistan':9,'Hong Kong':10,'West Indies':11,'South Africa':12,'United Arab Emirates':13,'Ireland':14,'New Zealand':15,'Netherlands':16,'Draw':17}}

matches.replace(encode, inplace=True)

matches.head(2)

#cities which are null then replace them with dubai(default)
matches[pd.isnull(matches['city'])]
matches['city'].fillna('Dubai',inplace=True)

#matches.describe()

#we maintain a dictionary for future reference mapping teams
dicVal = encode['winner']
matches = matches[['team1','team2','city','toss_decision','toss_winner','venue','winner']]

df = pd.DataFrame(matches)
df.describe()

#stats on the match winners and toss winners

temp1=df['toss_winner'].value_counts(sort=True)
temp2=df['winner'].value_counts(sort=True)

print('No of toss winners by each team')
for idx, val in temp1.iteritems():
   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))
print('No of match winners by each team')
for idx, val in temp2.iteritems():
   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))

#building predictive model
from sklearn.preprocessing import LabelEncoder
var_mod = ['city','toss_decision','venue']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes

# Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold  # For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    model.fit(data[predictors], data[outcome])

    predictions = model.predict(data[predictors])

    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
    return accuracy


#NaiveBayse Implementation
model = GaussianNB()
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)

#LogisticRegression
outcome_var=['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
model = LogisticRegression()
classification_model(model, df,predictor_var,outcome_var)
print("Random Forests")
#RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)

var1 = classification_model(model, df,predictor_var,outcome_var)
def acc():
    return var1*100

#DecisionTreeClassifier Implementaiton
model = DecisionTreeClassifier()
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)

#SupportVectorMachines
# model = SVC()
# outcome_var = ['winner']
# predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
# classification_model(model, df,predictor_var,outcome_var)


# #'team1', 'team2', 'venue', 'toss_winner','city','toss_decision'
# team1='PAK'
# team2='WID'
# toss_winner='WID'
# input=[dicVal[team1],dicVal[team2],'14',dicVal[toss_winner],'2','1']
# input = np.array(input).reshape((1, -1))
# output=model.predict(input)
# print(list(dicVal.keys())[list(dicVal.values()).index(output)]) #find key by value search output


imp_input = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(imp_input)


#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.13
# In conjunction with Tcl version 8.6
#    Apr 30, 2018 07:25:36 AM
import sys

# global team11, team22

try:
    from Tkinter import *

except ImportError:
    from tkinter import *

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

#import GUiTk2_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = Tk()
 #   GUiTk2_support.set_Tk_var()
    top = Cricket_Predition_Application (root)
  #  GUiTk2_support.init(root, top)
    root.mainloop()

w = None
def create_Cricket_Predition_Application(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = Toplevel (root)
 #   GUiTk2_support.set_Tk_var()
    top = Cricket_Predition_Application (w)
 #   GUiTk2_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Cricket_Predition_Application():
    global w
    w.destroy()
    w = None


class Stop (Exception):
    def __init__ (self):
        sys.tracebacklimit = 0

class Cricket_Predition_Application:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#d9d9d9' # X11 color: 'gray85'
        font10 = "-family {Segoe UI} -size 9 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"
        font12 = "-family Arial -size 15 -weight normal -slant roman "  \
            "-underline 0 -overstrike 0"
        font13 = "-family Arial -size 20 -weight normal -slant roman "  \
            "-underline 0 -overstrike 0"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("1001x558+223+27")
        top.title("Cricket Predition Application")
        top.configure(background="grey36")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")



        self.menubar = Menu(top,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        top.configure(menu = self.menubar)

        self.file = Menu(top,tearoff=0)
        self.menubar.add_cascade(menu=self.file,
                activebackground="#d9d9d9",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="File")
        self.file.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d8d8d8",
                font="TkMenuFont",
                foreground="#000000",
                label="New")
        self.file.add_command(
                activebackground="#4285f4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Open")
        self.file.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Save")
        self.file.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Save as...")
        self.file.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Close")
        self.file.add_separator(
                background="#d9d9d9")
        self.file.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font=font10,
                foreground="#000000",
                label="Exit")
        self.edit = Menu(top,tearoff=0)
        self.menubar.add_cascade(menu=self.edit,
                activebackground="#d9d9d9",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Edit")
        self.edit.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Undo")
        self.edit.add_separator(
                background="#d9d9d9")
        self.edit.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Cut")
        self.edit.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Copy")
        self.edit.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Paste")
        self.edit.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Delete")
        self.edit.add_command(
                activebackground="#d8d8d8",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Select All")
        self.window = Menu(top,tearoff=0)
        self.menubar.add_cascade(menu=self.window,
                activebackground="#d9d9d9",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Window")
        self.window.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Restore Default Layout")
        self.window.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Notifications")
        self.help = Menu(top,tearoff=0)
        self.menubar.add_cascade(menu=self.help,
                activebackground="#d9d9d9",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Help")
        self.help.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="_Get Started")
        self.help.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Submit Feedback...")
        self.help.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="Check for Updates...")
        self.help.add_command(
                activebackground="#4285F4",
                activeforeground="#000000",
                background="#d9d9d9",
                font="TkMenuFont",
                foreground="#000000",
                label="_About")


        self.Label1 = Label(top)
        self.Label1.place(relx=0.34, rely=0.02, height=61, width=314)
        self.Label1.configure(activebackground="#f9f9f9")
        self.Label1.configure(activeforeground="black")
        self.Label1.configure(background="#d9d9d9")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font=font13)
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(highlightbackground="#d9d9d9")
        self.Label1.configure(highlightcolor="black")
        self.Label1.configure(text='''Cricket Game Predictor''')
        self.Label1.configure(width=314)

        self.Frame1 = Frame(top)
        self.Frame1.place(relx=0.0, rely=0.3, relheight=0.65, relwidth=0.26)
        self.Frame1.configure(relief=GROOVE)
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief=GROOVE)
        self.Frame1.configure(background="grey36")
        self.Frame1.configure(width=265)

        self.Button2 =Button(self.Frame1)
        self.Button2.place(relx=0.0, rely=0.14, height=44, width=257)
        self.Button2.configure(activebackground="#d9d9d9")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(disabledforeground="#a3a3a3")
        self.Button2.configure(foreground="#000000")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(pady="0")
        self.Button2.configure(text='''Toss Statistics''')
        self.Button2.configure(width=257)

        self.Button1 = Button(self.Frame1)
        self.Button1.place(relx=0.0, rely=0.01, height=44, width=257)
        self.Button1.configure(activebackground="#d9d9d9")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="red")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Predict Match''')
        self.Button1.configure(width=257)




        self.Button2_1 = Button(self.Frame1)
        self.Button2_1.place(relx=0.0, rely=0.28, height=44, width=257)
        self.Button2_1.configure(activebackground="#d9d9d9")
        self.Button2_1.configure(activeforeground="#000000")
        self.Button2_1.configure(background="#d9d9d9")
        self.Button2_1.configure(disabledforeground="#a3a3a3")
        self.Button2_1.configure(foreground="#000000")
        self.Button2_1.configure(highlightbackground="#d9d9d9")
        self.Button2_1.configure(highlightcolor="black")
        self.Button2_1.configure(pady="0")
        self.Button2_1.configure(text='''Venue Statistics''')
        self.Button2_1.configure(width=257)

        self.Frame2 = Frame(top)
        self.Frame2.place(relx=0.28, rely=0.3, relheight=0.65, relwidth=0.69)
        self.Frame2.configure(relief=GROOVE)
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief=GROOVE)
        self.Frame2.configure(background="grey36")
        self.Frame2.configure(width=695)

        self.Label2 = Label(self.Frame2)
        self.Label2.place(relx=0.04, rely=0.14, height=25, width=100)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="ivory3")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font=font12)
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Team A:''')

        # self.Label2 = Label(self.Frame2)
        # self.Label2.place(relx=0.55, rely=0.14, height=25, width=130)
        # self.Label2.configure(activebackground="#f9f9f9")
        # self.Label2.configure(activeforeground="black")
        # self.Label2.configure(background="ivory3")
        # self.Label2.configure(disabledforeground="#a3a3a3")
        # self.Label2.configure(font=font12)
        # self.Label2.configure(foreground="#000000")
        # self.Label2.configure(highlightbackground="#d9d9d9")
        # self.Label2.configure(highlightcolor="black")
        # self.Label2.configure(text='''TossDecision:''')
        #
        # self.TCombobox_tossDecision = ttk.Combobox(self.Frame2)
        # self.TCombobox_tossDecision.place(relx=0.75, rely=0.14, relheight=0.06
        #                                   , relwidth=0.20)
        # self.TCombobox_tossDecision['values'] = ('Bowl', 'Bat')

        # self.Label2_city = Label(self.Frame2)
        # self.Label2_city.place(relx=0.55, rely=0.27, height=22, width=80)
        # self.Label2_city.configure(activebackground="#f9f9f9")
        # self.Label2_city.configure(activeforeground="black")
        # self.Label2_city.configure(background="ivory3")
        # self.Label2_city.configure(disabledforeground="#a3a3a3")
        # self.Label2_city.configure(font=font12)
        # self.Label2_city.configure(foreground="#000000")
        # self.Label2_city.configure(highlightbackground="#d9d9d9")
        # self.Label2_city.configure(highlightcolor="black")
        # self.Label2_city.configure(text='''City:''')

        # self.Entry1_city = Entry(self.Frame2)
        # self.Entry1_city.place(relx=0.75, rely=0.27, height=20, relwidth=0.20)
        # # self.Entry1_city.configure(textvariable=mystring)
        # self.Entry1_city.configure(background="white")
        # self.Entry1_city.configure(disabledforeground="#a3a3a3")
        # self.Entry1_city.configure(font="TkFixedFont")
        # self.Entry1_city.configure(foreground="#000000")
        # self.Entry1_city.configure(insertbackground="black")
        # self.Entry1_city.configure(width=194)

        self.Label2_1 = Label(self.Frame2)
        self.Label2_1.place(relx=0.04, rely=0.28, height=21, width=94)
        self.Label2_1.configure(activebackground="#f9f9f9")
        self.Label2_1.configure(activeforeground="black")
        self.Label2_1.configure(background="#d9d9d9")
        self.Label2_1.configure(disabledforeground="#a3a3a3")
        self.Label2_1.configure(font=font12)
        self.Label2_1.configure(foreground="#000000")
        self.Label2_1.configure(highlightbackground="#d9d9d9")
        self.Label2_1.configure(highlightcolor="black")
        self.Label2_1.configure(text='''Team B:''')

        self.Label2_2 = Label(self.Frame2)
        self.Label2_2.place(relx=0.04, rely=0.42, height=21, width=102)
        self.Label2_2.configure(activebackground="#f9f9f9")
        self.Label2_2.configure(activeforeground="black")
        self.Label2_2.configure(background="#d9d9d9")
        self.Label2_2.configure(disabledforeground="#a3a3a3")
        self.Label2_2.configure(font=font12)
        self.Label2_2.configure(foreground="#000000")
        self.Label2_2.configure(highlightbackground="#d9d9d9")
        self.Label2_2.configure(highlightcolor="black")
        self.Label2_2.configure(text='''TossWinner''')


       # mystring = StringVar()

        # def getvalue():

        # mystring2 = StringVar()
        #
        # #     print(mystring.get())
        # def onAcc():
        #     return mystring2.get()

        mystring = StringVar()
        #     print(mystring.get())
        def onchick():
            return mystring.get()


        # def getvalue():
        #     print(mystring.get())

        self.Entry1 = Entry(self.Frame2)
        self.Entry1.place(relx=0.19, rely=0.41,height=20, relwidth=0.28)
        self.Entry1.configure(textvariable=mystring)
        self.Entry1.configure(background="white")
        self.Entry1.configure(disabledforeground="#a3a3a3")
        self.Entry1.configure(font="TkFixedFont")
        self.Entry1.configure(foreground="#000000")
        self.Entry1.configure(insertbackground="black")
        self.Entry1.configure(width=194)

        #print(mystring)
        self.Label2_3 = Label(self.Frame2)
        self.Label2_3.place(relx=0.04, rely=0.58, height=21, width=94)
        self.Label2_3.configure(activebackground="#f9f9f9")
        self.Label2_3.configure(activeforeground="black")
        self.Label2_3.configure(background="#d9d9d9")
        self.Label2_3.configure(disabledforeground="#a3a3a3")
        self.Label2_3.configure(font=font12)
        self.Label2_3.configure(foreground="#000000")
        self.Label2_3.configure(highlightbackground="#d9d9d9")
        self.Label2_3.configure(highlightcolor="black")
        self.Label2_3.configure(text='''Venue:''')

        self.TCombobox1_ve = ttk.Combobox(self.Frame2)
        self.TCombobox1_ve.place(relx=0.19, rely=0.58,height=20, relwidth=0.28)

        # def justamethod2(event):
        #     print(self.TCombobox1_6.get())

        self.TCombobox1_ve['values'] = ('Sydney Cricket Ground', 'Bellerive Oval', 'Melbrune Cricket Ground', 'Sydney Cricket Ground', 'Manuka Oval',
                                       'Greater Noida Sports Complex Ground', 'Sharjah Cricket Stadium', 'Sheikh Zayed Stadium', 'Harare Sports Club',
                                       'Queens Sports Club', 'Lords', 'Dubai International Cricket Stadium')



        # self.Entry1_4 = Entry(self.Frame2)
        # self.Entry1_4.place(relx=0.19, rely=0.89,height=20, relwidth=0.28)
        # self.Entry1_4.configure(background="white")
        # self.Entry1_4.configure(disabledforeground="#a3a3a3")
        # self.Entry1_4.configure(font="TkFixedFont")
        # self.Entry1_4.configure(foreground="#000000")
        # self.Entry1_4.configure(highlightbackground="#d9d9d9")
        # self.Entry1_4.configure(highlightcolor="black")
        # self.Entry1_4.configure(insertbackground="black")
        # self.Entry1_4.configure(selectbackground="#c4c4c4")
        # self.Entry1_4.configure(selectforeground="black")
        # self.Entry1_4.configure(width=194)

        self.Button1 = Button(self.Frame2)
        self.Button1.place(relx=0.29, rely=0.74, height=44, width=127)
        self.Button1.configure(activebackground="#d9d9d9")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Predict''')
        self.Button1.configure(width=127)



        outp = StringVar()
        def uifun(outp):
            self.Entry2 = Entry(self.Frame2)
            self.Entry2.place(relx=0.58, rely=0.68, height=50, relwidth=0.32)
            self.Entry2.configure(textvariable=outp)
            self.Entry2.insert(0, outp)
            self.Entry2.configure(background="white")
            self.Entry2.configure(disabledforeground="#a3a3a3")
            self.Entry2.configure(font="TkFixedFont")
            self.Entry2.configure(foreground="#000000")
            self.Entry2.configure(insertbackground="black")
            self.Entry2.configure(width=224)


        def uiTosswin():

            stt = StringVar()
            self.Entry1 = Entry(self.Frame2)
            self.Entry1.place(relx=0.19, rely=0.41, height=20, relwidth=0.28)
            self.Entry1.configure(background="white")
            self.Entry1.configure(disabledforeground="#a3a3a3")
            self.Entry1.configure(font="TkFixedFont")
            self.Entry1.configure(foreground="#000000")
            self.Entry1.configure(insertbackground="black")
            self.Entry1.configure(width=194)
            return self.Entry1.configure(self, textvariable=stt)

        def uiAccu():
            self.Entry3 = Entry(self.Frame2)
            self.Entry3.place(relx=0.71, rely=0.42, height=30, relwidth=0.25)
            self.Entry3.configure(background="white")

            # acc1 = acc()
            str1 = StringVar()
            str1 = acc()

            self.Entry3.configure(textvariable=str1)
            self.Entry3.configure(disabledforeground="#a3a3a3")
            self.Entry3.insert(0, str1)
            self.Entry3.configure(font="TkFixedFont")
            self.Entry3.configure(foreground="#000000")
            self.Entry3.configure(insertbackground="black")
            self.Entry3.configure(width=104)

        f = False

        def onclick():
            global f
            f = True
            tossTeam = StringVar()

            def justamethod(event):
                return self.TCombobox1.get()

            comb1 = justamethod(self)
            self.TCombobox1.bind("<<ComboboxSelected>>", justamethod(self))

            def justamethod2(event):
                 return self.TCombobox1_6.get()

            comb2 = justamethod2(self)

            self.TCombobox1_6.bind("<<ComboboxSelected>>", justamethod2(self))
            tossTeam = onchick()

            # 'team1', 'team2', 'venue', 'toss_winner','city','toss_decision'
            team1 = comb1
            team2 = comb2
            if comb1 == "" and comb2 == "":
                messagebox.showerror("Error", "Team Names Cannot be Null")
                raise Stop()
            print( comb1)
            if (comb1 == comb2):
                #messagebox.showerror("Error", "Team Names Should be Different")
                messagebox.showerror("Error", "Team Names Should be different")
                raise Stop ()
            #sys.exit()
            print(comb2)

            toss_winner = tossTeam
            print("Toss Winner")
            print(tossTeam)
            if toss_winner not in (comb1,comb2):
                messagebox.showerror("Error", "Wrong toss Winner")
                raise Stop ()



            input = [dicVal[team1], dicVal[team2], '14', dicVal[toss_winner], '2', '1']
            input = np.array(input).reshape((1, -1))
            output = model.predict(input)
            outp = list(dicVal.keys())[list(dicVal.values()).index(output)]# find key by value search output

            uifun(outp)
            uiAccu()
            bb = False

            def tossStats():
                global bb
                bb = True
                mlt.style.use('fivethirtyeight')
                df_fil = df[df['toss_winner'] == df['winner']]
                slices = [len(df_fil), (577 - len(df_fil))]
                mlt.pie(slices, labels=['Toss & Loss', 'Toss & Win'], startangle=90, shadow=True, explode=(0, 0),
                        autopct='%1.1f%%', colors=['r', 'g'])
                fig = mlt.gcf()
                fig.set_size_inches(5, 5)
                mlt.show()

            self.Button2.config(command=tossStats)

            ff = False

            def venueStats():
                global ff
                ff = True
                import seaborn as sns
                team1 = dicVal[comb1]
                team2 = dicVal[comb2]
                mtemp = matches[((matches['team1'] == team1) | (matches['team2'] == team1)) & (
                    (matches['team1'] == team2) | (matches['team2'] == team2))]
                sns.countplot(x='venue', hue='winner', data=mtemp, palette='Set3')
                mlt.xticks(rotation='vertical')
                leg = mlt.legend(loc='upper right')
                fig = mlt.gcf()
                fig.set_size_inches(10, 6)
                mlt.show()

            self.Button2_1.config(command=venueStats)

        self.Button1.config(command=onclick)

        self.Label2 = Label(self.Frame2)
        self.Label2.place(relx=0.66, rely=0.61, height=21, width=94)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font=font12)
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Winner!''')

        self.Entry2 = Entry(self.Frame2)
        self.Entry2.place(relx=0.58, rely=0.68, height=50, relwidth=0.32)
        self.Entry2.configure(background="white")
        self.Entry2.configure(disabledforeground="#a3a3a3")
        self.Entry2.configure(font="TkFixedFont")
        self.Entry2.configure(foreground="#000000")
        self.Entry2.configure(insertbackground="black")
        self.Entry2.configure(width=224)

        self.Label2_5 = Label(self.Frame2)
        self.Label2_5.place(relx=0.55, rely=0.42, height=21, width=94)
        self.Label2_5.configure(activebackground="#f9f9f9")
        self.Label2_5.configure(activeforeground="black")
        self.Label2_5.configure(background="#d9d9d9")
        self.Label2_5.configure(disabledforeground="#a3a3a3")
        self.Label2_5.configure(font=font12)
        self.Label2_5.configure(foreground="#000000")
        self.Label2_5.configure(highlightbackground="#d9d9d9")
        self.Label2_5.configure(highlightcolor="black")
        self.Label2_5.configure(text='''Accuracy:''')

        self.Entry3 = Entry(self.Frame2)
        self.Entry3.place(relx=0.71, rely=0.42,height=30, relwidth=0.25)
        self.Entry3.configure(background="white")

        # #acc1 = acc()
        # str1 = StringVar()
        # str1= acc()

        # self.Entry3.configure(textvariable=str1)
        self.Entry3.configure(disabledforeground="#a3a3a3")
        # self.Entry3.insert(0,str1)
        self.Entry3.configure(font="TkFixedFont")
        self.Entry3.configure(foreground="#000000")
        self.Entry3.configure(insertbackground="black")
        self.Entry3.configure(width=104)

        self.TCombobox1 = ttk.Combobox(self.Frame2)
        self.TCombobox1.place(relx=0.19, rely=0.14, relheight=0.06
                , relwidth=0.28)




        self.TCombobox1['values'] = ('Pakistan','England','India','Australia','Zimbabwe',
                 'Bangladesh','Sri Lanka','Scotland','Afghanistan',
                 'Hong Kong','West Indies','South Africa','United Arab Emirates','Ireland','New Zealand','Netherlands')
        #self.TCombobox1_6.bind("<<ComboboxSelected>>", justamethod2(self))
        #self.TCombobox1.current(0)

        # def justamethod(event):
        #     print(self.TCombobox1.get())
        #
        # self.TCombobox1.bind("<<ComboboxSelected>>", justamethod(self))

        # vall = justamethod(self)
        # print(vall)

        self.TCombobox1.configure(width=193)
        self.TCombobox1.configure(takefocus="")


        self.TCombobox1_6 = ttk.Combobox(self.Frame2)
        self.TCombobox1_6.place(relx=0.19, rely=0.28, relheight=0.06
                , relwidth=0.28)

        # def justamethod2(event):
        #     print(self.TCombobox1_6.get())

        self.TCombobox1_6['values'] = ('Pakistan', 'England', 'India', 'Australia', 'Zimbabwe',
                                     'Bangladesh', 'Sri Lanka', 'Scotland', 'Afghanistan',
                                     'Hong Kong', 'West Indies', 'South Africa', 'United Arab Emirates', 'Ireland',
                                     'New Zealand', 'Netherlands')

        # self.TCombobox1_6.bind("<<ComboboxSelected>>", justamethod2(self))
       #self.TCombobox1_6.configure(textvariable=GUiTk2_support.combobox)
        self.TCombobox1_6.configure(takefocus="")

    @staticmethod
    def popup1(event):
        Popupmenu1 = Menu(root, tearoff=0)
        Popupmenu1.configure(activebackground="#f9f9f9")
        Popupmenu1.configure(activeborderwidth="1")
        Popupmenu1.configure(activeforeground="black")
        Popupmenu1.configure(background="#d9d9d9")
        Popupmenu1.configure(borderwidth="1")
        Popupmenu1.configure(disabledforeground="#a3a3a3")
        Popupmenu1.configure(font="{Segoe UI} 9")
        Popupmenu1.configure(foreground="black")
        Popupmenu1.post(event.x_root, event.y_root)



if __name__ == '__main__':
    vp_start_gui()





































