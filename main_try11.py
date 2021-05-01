##################################################
### Created by Lilian Sao de Rivera
### Project Name : The economics of happiness
### Date 04/23/2017
### Data Mining
##################################################

import sys

#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

from scipy import interp
from itertools import cycle

from PyQt5 import QtGui
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox,QMenu

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
# from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import random
import seaborn as sns

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
# os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------


#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'

class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Random Forest Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Random Forest Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature=[]

        for i in range(12):
            self.feature.append(QCheckBox(features_list[i],self))

        for i in self.feature:
            i.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblNumberesti = QLabel('Number of estimators :')
        self.lblNumberesti.adjustSize()

        self.txtNumberesti = QLineEdit(self)
        self.txtNumberesti.setText("10")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.btnRoc_Execute = QPushButton("Plot ROC")
        self.btnRoc_Execute.clicked.connect(self.roc_update)

        self.groupBox1Layout.addWidget(self.feature[0],0,0)
        self.groupBox1Layout.addWidget(self.feature[1],0,1)
        self.groupBox1Layout.addWidget(self.feature[2],1,0)
        self.groupBox1Layout.addWidget(self.feature[3],1,1)
        self.groupBox1Layout.addWidget(self.feature[4],2,0)
        self.groupBox1Layout.addWidget(self.feature[5],2,1)
        self.groupBox1Layout.addWidget(self.feature[6],3,0)
        self.groupBox1Layout.addWidget(self.feature[7],3,1)
        self.groupBox1Layout.addWidget(self.feature[8],4,0)
        self.groupBox1Layout.addWidget(self.feature[9],4,1)
        self.groupBox1Layout.addWidget(self.feature[10],5,0)
        self.groupBox1Layout.addWidget(self.feature[11],5,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest,7,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,7,1)
        self.groupBox1Layout.addWidget(self.lblNumberesti,8,0)
        self.groupBox1Layout.addWidget(self.txtNumberesti,8,1)
        self.groupBox1Layout.addWidget(self.btnExecute,9,0)
        self.groupBox1Layout.addWidget(self.btnRoc_Execute,9,1)

        self.groupBox2 = QGroupBox('Results from the Gini model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults1 = QLabel('Results :')
        self.lblResults1.adjustSize()
        self.txtResults1 = QPlainTextEdit()
        self.lblAccuracy1 = QLabel('Accuracy :')
        self.txtAccuracy1 = QLineEdit()
        self.lblRoc_auc1 = QLabel('ROC_AUC :')
        self.txtRoc_auc1 = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults1)
        self.groupBox2Layout.addWidget(self.txtResults1)
        self.groupBox2Layout.addWidget(self.lblAccuracy1)
        self.groupBox2Layout.addWidget(self.txtAccuracy1)
        self.groupBox2Layout.addWidget(self.lblRoc_auc1)
        self.groupBox2Layout.addWidget(self.txtRoc_auc1)

        self.groupBox3 = QGroupBox('Results from the Entropy model')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        self.lblResults2 = QLabel('Results :')
        self.lblResults2.adjustSize()
        self.txtResults2 = QPlainTextEdit()
        self.lblAccuracy2 = QLabel('Accuracy :')
        self.txtAccuracy2 = QLineEdit()
        self.lblRoc_auc2 = QLabel('ROC_AUC :')
        self.txtRoc_auc2 = QLineEdit()

        self.groupBox3Layout.addWidget(self.lblResults2)
        self.groupBox3Layout.addWidget(self.txtResults2)
        self.groupBox3Layout.addWidget(self.lblAccuracy2)
        self.groupBox3Layout.addWidget(self.txtAccuracy2)
        self.groupBox3Layout.addWidget(self.lblRoc_auc2)
        self.groupBox3Layout.addWidget(self.txtRoc_auc2)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix - Gini model
        #::--------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix (Gini model):')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        #::--------------------------------------
        # Graphic 2 : Confusion Matrix - Entropy model
        #::--------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes1=[self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Confusion Matrix (Entropy model):')
        self.groupBoxG2Layout= QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,1)
        self.layout.addWidget(self.groupBoxG2,0,2)
        self.layout.addWidget(self.groupBox3,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''

        # processing the parameters

        self.current_features = pd.DataFrame([])
        for i in range(12):
            if self.feature[i].isChecked():
                if len(self.current_features)==0:
                    self.current_features = data[features_list[i]]
                else:
                    self.current_features = pd.concat([self.current_features, data[features_list[i]]],axis=1)


        vtest_per = float(self.txtPercentTest.text())
        n_esti = int(self.txtNumberesti.text())
        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax2.clear()

        self.txtResults1.clear()
        self.txtResults1.setUndoRedoEnabled(False)

        self.txtResults2.clear()
        self.txtResults2.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_data1=pd.get_dummies(self.current_features.loc[:, self.current_features.dtypes == 'object'])
        X_data2=self.current_features.loc[:, self.current_features.dtypes != 'object']

        X_data3=pd.concat([X_data1,X_data2],axis=1)

        X = X_data3.values

        y=data.iloc[:,-1]

        class_le = LabelEncoder()

        # fit and transform the class

        y = class_le.fit_transform(y)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vtest_per, random_state=100)

        # perform training with entropy.
        # Decision tree with entropy

        #::------------------------------------
        ##  Model 1 - gini model:

        #::------------------------------------

        #specify random forest classifier
        self.clf_rf_gini = RandomForestClassifier(n_estimators=n_esti,criterion='gini', random_state=100)

        # perform training
        self.clf_rf_gini.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred_gini = self.clf_rf_gini.predict(X_test)
        y_pred_score_gini = self.clf_rf_gini.predict_proba(X_test)

        # confusion matrix for RandomForest
        conf_matrix_gini = confusion_matrix(y_test, y_pred_gini)

        # clasification report

        self.class_rep_gini = classification_report(y_test, y_pred_gini)
        self.txtResults1.appendPlainText(self.class_rep_gini)

        # accuracy score

        self.accuracy_score_gini = accuracy_score(y_test, y_pred_gini) * 100
        self.txtAccuracy1.setText(str(self.accuracy_score_gini))

        # ROC-AUC
        self.rocauc_score_gini = roc_auc_score(y_test,y_pred_score_gini[:,1]) * 100
        self.txtRoc_auc1.setText(str(self.rocauc_score_gini))

        self.fpr_gini,self.tpr_gini, _ = roc_curve(y_test,  y_pred_score_gini[:,1])
        self.auc_gini = roc_auc_score(y_test, y_pred_score_gini[:,1])

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------

        df_cm_gini = pd.DataFrame(conf_matrix_gini, index=class_names, columns=class_names )


        hm1 = sns.heatmap(df_cm_gini, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm_gini.columns, xticklabels=df_cm_gini.columns,ax=self.ax1)

        hm1.yaxis.set_ticklabels(hm1.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        hm1.xaxis.set_ticklabels(hm1.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=10)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()


        #::------------------------------------
        ##  Model 2 - entropy model:

        #::------------------------------------

        #specify random forest classifier
        self.clf_rf_entropy = RandomForestClassifier(n_estimators=n_esti,criterion='entropy', random_state=100)

        # perform training
        self.clf_rf_entropy.fit(X_train, y_train)

        # predicton on test using all features
        y_pred_entropy = self.clf_rf_entropy.predict(X_test)
        y_pred_score_entropy = self.clf_rf_entropy.predict_proba(X_test)


        # confusion matrix for RandomForest
        conf_matrix_entropy = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.class_rep_entropy = classification_report(y_test, y_pred_entropy)
        self.txtResults2.appendPlainText(self.class_rep_entropy)

        # accuracy score

        self.accuracy_score_entropy = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy2.setText(str(self.accuracy_score_entropy))

        # ROC-AUC
        self.rocauc_score_entropy = roc_auc_score(y_test,y_pred_score_entropy[:,1]) * 100
        self.txtRoc_auc2.setText(str(self.rocauc_score_entropy))

        self.fpr_entropy,self.tpr_entropy, _ = roc_curve(y_test,  y_pred_score_entropy[:,1])
        self.auc_entropy = roc_auc_score(y_test, y_pred_score_entropy[:,1])

        #::------------------------------------
        ##  Ghaph2 :
        ##  Confusion Matrix - entropy model
        #::------------------------------------

        df_cm_entropy = pd.DataFrame(conf_matrix_entropy, index=class_names, columns=class_names )

        hm2 = sns.heatmap(df_cm_entropy, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm_entropy.columns, xticklabels=df_cm_entropy.columns,ax=self.ax2)

        hm2.yaxis.set_ticklabels(hm2.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        hm2.xaxis.set_ticklabels(hm2.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=10)
        self.ax2.set_xlabel('Predicted label')
        self.ax2.set_ylabel('True label')
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

    def roc_update(self):
        # gini model
        dialog = CanvasWindow(self)

        dialog.m.plot()
        # roc_gini=plot_roc_curve(self.clf_df_gini,self.X_test,self.y_test,ax=dialog.m.ax)
        dialog.m.ax.plot(self.fpr_gini, self.tpr_gini, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % self.auc_gini)
        dialog.m.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        dialog.m.ax.set_title('ROC of Gini model')
        dialog.m.ax.set_xlim([0.0, 1.0])
        dialog.m.ax.set_ylim([0.0, 1.05])
        dialog.m.ax.set_xlabel("False Positive Rate")
        dialog.m.ax.set_ylabel("True Positive Rate")
        dialog.m.ax.legend(loc="lower right")
        dialog.m.draw()
        dialog.show()

        # entropy model
        dialog = CanvasWindow(self)
        dialog.m.plot()
        # roc_entropy=plot_roc_curve(self.clf_df_entropy,self.X_test1,self.y_test1,ax=dialog.m.ax)
        dialog.m.ax.plot(self.fpr_entropy, self.tpr_entropy, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % self.auc_entropy)
        dialog.m.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        dialog.m.ax.set_title('ROC of Entropy model')
        dialog.m.ax.set_xlim([0.0, 1.0])
        dialog.m.ax.set_ylim([0.0, 1.05])
        dialog.m.ax.set_xlabel("False Positive Rate")
        dialog.m.ax.set_ylabel("True Positive Rate")
        dialog.m.ax.legend(loc="lower right")
        dialog.m.draw()
        dialog.show()

class DecisionTree(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.Title = "Decision Tree Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Decision Tree Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature=[]

        for i in range(12):
            self.feature.append(QCheckBox(features_list[i],self))

        for i in self.feature:
            i.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('Maximun Depth :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute DT")
        self.btnExecute.clicked.connect(self.update)

        self.btnRoc_Execute = QPushButton("Plot ROC")
        self.btnRoc_Execute.clicked.connect(self.roc_update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)

        self.groupBox1Layout.addWidget(self.feature[0],0,0)
        self.groupBox1Layout.addWidget(self.feature[1],0,1)
        self.groupBox1Layout.addWidget(self.feature[2],1,0)
        self.groupBox1Layout.addWidget(self.feature[3],1,1)
        self.groupBox1Layout.addWidget(self.feature[4],2,0)
        self.groupBox1Layout.addWidget(self.feature[5],2,1)
        self.groupBox1Layout.addWidget(self.feature[6],3,0)
        self.groupBox1Layout.addWidget(self.feature[7],3,1)
        self.groupBox1Layout.addWidget(self.feature[8],4,0)
        self.groupBox1Layout.addWidget(self.feature[9],4,1)
        self.groupBox1Layout.addWidget(self.feature[10],5,0)
        self.groupBox1Layout.addWidget(self.feature[11],5,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest,7,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,7,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,8,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,8,1)
        self.groupBox1Layout.addWidget(self.btnExecute,9,0)
        self.groupBox1Layout.addWidget(self.btnRoc_Execute,9,1)
        self.groupBox1Layout.addWidget(self.btnDTFigure,10,1)

        self.groupBox2 = QGroupBox('Results from the Gini model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults1 = QLabel('Results :')
        self.lblResults1.adjustSize()
        self.txtResults1 = QPlainTextEdit()
        self.lblAccuracy1 = QLabel('Accuracy :')
        self.txtAccuracy1 = QLineEdit()
        self.lblRoc_auc1 = QLabel('ROC_AUC :')
        self.txtRoc_auc1 = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults1)
        self.groupBox2Layout.addWidget(self.txtResults1)
        self.groupBox2Layout.addWidget(self.lblAccuracy1)
        self.groupBox2Layout.addWidget(self.txtAccuracy1)
        self.groupBox2Layout.addWidget(self.lblRoc_auc1)
        self.groupBox2Layout.addWidget(self.txtRoc_auc1)

        self.groupBox3 = QGroupBox('Results from the Entropy model')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        self.lblResults2 = QLabel('Results :')
        self.lblResults2.adjustSize()
        self.txtResults2 = QPlainTextEdit()
        self.lblAccuracy2 = QLabel('Accuracy :')
        self.txtAccuracy2 = QLineEdit()
        self.lblRoc_auc2 = QLabel('ROC_AUC :')
        self.txtRoc_auc2 = QLineEdit()

        self.groupBox3Layout.addWidget(self.lblResults2)
        self.groupBox3Layout.addWidget(self.txtResults2)
        self.groupBox3Layout.addWidget(self.lblAccuracy2)
        self.groupBox3Layout.addWidget(self.txtAccuracy2)
        self.groupBox3Layout.addWidget(self.lblRoc_auc2)
        self.groupBox3Layout.addWidget(self.txtRoc_auc2)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix - Gini model
        #::--------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix (Gini model):')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        #::--------------------------------------
        # Graphic 2 : Confusion Matrix - Entropy model
        #::--------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes1=[self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Confusion Matrix (Entropy model):')
        self.groupBoxG2Layout= QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,1)
        self.layout.addWidget(self.groupBoxG2,0,2)
        self.layout.addWidget(self.groupBox3,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''

        # processing the parameters

        self.current_features = pd.DataFrame([])
        for i in range(12):
            if self.feature[i].isChecked():
                if len(self.current_features)==0:
                    self.current_features = data[features_list[i]]
                else:
                    self.current_features = pd.concat([self.current_features, data[features_list[i]]],axis=1)

        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = float(self.txtMaxDepth.text())
        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax2.clear()

        self.txtResults1.clear()
        self.txtResults1.setUndoRedoEnabled(False)

        self.txtResults2.clear()
        self.txtResults2.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        # label encoding the categorical data
        class_le1 = LabelEncoder()
        features_list1 = self.current_features.loc[:, self.current_features.dtypes == 'object'].columns
        for i in features_list1:
            self.current_features[i]=class_le1.fit_transform(self.current_features[i])

        X = self.current_features.values

        y=data.iloc[:,-1]

        # label encoding the target
        class_le2 = LabelEncoder()
        y = class_le2.fit_transform(y)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vtest_per, random_state=100)

        # self.X_test1=X_test
        # self.y_test1=y_test
        # perform training with entropy.
        #::------------------------------------
        ##  Model 1 - gini model:

        #::------------------------------------

        #specify random forest classifier
        self.clf_df_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=vmax_depth, min_samples_leaf=5)

        # perform training
        self.clf_df_gini.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred_gini = self.clf_df_gini.predict(X_test)
        y_pred_score_gini = self.clf_df_gini.predict_proba(X_test)


        # confusion matrix for RandomForest
        conf_matrix_gini = confusion_matrix(y_test, y_pred_gini)

        # clasification report

        self.class_rep_gini = classification_report(y_test, y_pred_gini)
        self.txtResults1.appendPlainText(self.class_rep_gini)

        # accuracy score

        self.accuracy_score_gini = accuracy_score(y_test, y_pred_gini) * 100
        self.txtAccuracy1.setText(str(self.accuracy_score_gini))

        # ROC-AUC
        self.rocauc_score_gini = roc_auc_score(y_test,y_pred_score_gini[:,1]) * 100
        self.txtRoc_auc1.setText(str(self.rocauc_score_gini))

        self.fpr_gini,self.tpr_gini, _ = roc_curve(y_test,  y_pred_score_gini[:,1])
        self.auc_gini = roc_auc_score(y_test, y_pred_score_gini[:,1])

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------

        df_cm_gini = pd.DataFrame(conf_matrix_gini, index=class_names, columns=class_names )

        hm1 = sns.heatmap(df_cm_gini, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm_gini.columns, xticklabels=df_cm_gini.columns,ax=self.ax1)

        hm1.yaxis.set_ticklabels(hm1.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        hm1.xaxis.set_ticklabels(hm1.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=10)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        #::------------------------------------
        ##  Model 2 - entropy model:

        #::------------------------------------

        #specify random forest classifier
        self.clf_df_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=vmax_depth, min_samples_leaf=5)

        # perform training
        self.clf_df_entropy.fit(X_train, y_train)

        # predicton on test using all features
        y_pred_entropy = self.clf_df_entropy.predict(X_test)
        y_pred_score_entropy = self.clf_df_entropy.predict_proba(X_test)


        # confusion matrix for RandomForest
        conf_matrix_entropy = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.class_rep_entropy = classification_report(y_test, y_pred_entropy)
        self.txtResults2.appendPlainText(self.class_rep_entropy)

        # accuracy score

        self.accuracy_score_entropy = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy2.setText(str(self.accuracy_score_entropy))

        # ROC-AUC
        self.rocauc_score_entropy = roc_auc_score(y_test,y_pred_score_entropy[:,1]) * 100
        self.txtRoc_auc2.setText(str(self.rocauc_score_entropy))

        self.fpr_entropy,self.tpr_entropy, _ = roc_curve(y_test,  y_pred_score_entropy[:,1])
        self.auc_entropy = roc_auc_score(y_test, y_pred_score_entropy[:,1])

        #::------------------------------------
        ##  Ghaph2 :
        ##  Confusion Matrix - entropy model
        #::------------------------------------

        df_cm_entropy = pd.DataFrame(conf_matrix_entropy, index=class_names, columns=class_names )

        hm2 = sns.heatmap(df_cm_entropy, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm_entropy.columns, xticklabels=df_cm_entropy.columns,ax=self.ax2)

        hm2.yaxis.set_ticklabels(hm2.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        hm2.xaxis.set_ticklabels(hm2.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=10)
        self.ax2.set_xlabel('Predicted label')
        self.ax2.set_ylabel('True label')
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

    def roc_update(self):
        # gini model
        dialog = CanvasWindow(self)

        dialog.m.plot()
        # roc_gini=plot_roc_curve(self.clf_df_gini,self.X_test,self.y_test,ax=dialog.m.ax)
        dialog.m.ax.plot(self.fpr_gini, self.tpr_gini, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % self.auc_gini)
        dialog.m.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        dialog.m.ax.set_title('ROC of Gini model')
        dialog.m.ax.set_xlim([0.0, 1.0])
        dialog.m.ax.set_ylim([0.0, 1.05])
        dialog.m.ax.set_xlabel("False Positive Rate")
        dialog.m.ax.set_ylabel("True Positive Rate")
        dialog.m.ax.legend(loc="lower right")
        dialog.m.draw()
        dialog.show()

        # entropy model
        dialog = CanvasWindow(self)
        dialog.m.plot()
        # roc_entropy=plot_roc_curve(self.clf_df_entropy,self.X_test1,self.y_test1,ax=dialog.m.ax)
        dialog.m.ax.plot(self.fpr_entropy, self.tpr_entropy, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % self.auc_entropy)
        dialog.m.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        dialog.m.ax.set_title('ROC of Entropy model')
        dialog.m.ax.set_xlim([0.0, 1.0])
        dialog.m.ax.set_ylim([0.0, 1.05])
        dialog.m.ax.set_xlabel("False Positive Rate")
        dialog.m.ax.set_ylabel("True Positive Rate")
        dialog.m.ax.legend(loc="lower right")
        dialog.m.draw()
        dialog.show()

    def view_tree(self):
        '''
        Executes the graphviz to create a tree view of the information
         then it presents the graphic in a pdf formt using webbrowser
        :return:None
        '''

        # produces pdf results of gini model
        dot_data1 = export_graphviz(self.clf_df_gini, filled=True, rounded=True, class_names=class_names,
                                    feature_names=self.current_features.columns, out_file=None)


        graph = graph_from_dot_data(dot_data1)
        graph.write_pdf("decision_tree_gini.pdf")
        webbrowser.open_new(r'decision_tree_gini.pdf')

        # produces pdf results of entropy model
        dot_data2 = export_graphviz(self.clf_df_entropy, filled=True, rounded=True, class_names=class_names,
                                    feature_names=self.current_features.columns, out_file=None)


        graph = graph_from_dot_data(dot_data2)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r'decision_tree_entropy.pdf')


class SupportVector(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(SupportVector, self).__init__()
        self.Title = "Support Vector Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Support Vector Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
        self.feature=[]

        for i in range(12):
            self.feature.append(QCheckBox(features_list[i],self))

        for i in self.feature:
            i.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblKernelType = QLabel('kernel :')
        self.lblKernelType.adjustSize()

        self.txtKernelType = QLineEdit(self)
        self.txtKernelType.setText("rbf")

        self.btnExecute = QPushButton("Execute SVC")
        self.btnExecute.clicked.connect(self.update)

        self.btnRoc_Execute = QPushButton("Plot ROC")
        self.btnRoc_Execute.clicked.connect(self.roc_update)

        self.groupBox1Layout.addWidget(self.feature[0],0,0)
        self.groupBox1Layout.addWidget(self.feature[1],0,1)
        self.groupBox1Layout.addWidget(self.feature[2],1,0)
        self.groupBox1Layout.addWidget(self.feature[3],1,1)
        self.groupBox1Layout.addWidget(self.feature[4],2,0)
        self.groupBox1Layout.addWidget(self.feature[5],2,1)
        self.groupBox1Layout.addWidget(self.feature[6],3,0)
        self.groupBox1Layout.addWidget(self.feature[7],3,1)
        self.groupBox1Layout.addWidget(self.feature[8],4,0)
        self.groupBox1Layout.addWidget(self.feature[9],4,1)
        self.groupBox1Layout.addWidget(self.feature[10],5,0)
        self.groupBox1Layout.addWidget(self.feature[11],5,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest,7,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,7,1)
        self.groupBox1Layout.addWidget(self.lblKernelType,8,0)
        self.groupBox1Layout.addWidget(self.txtKernelType,8,1)
        self.groupBox1Layout.addWidget(self.btnExecute,9,0)
        self.groupBox1Layout.addWidget(self.btnRoc_Execute,10,0)

        self.groupBox2 = QGroupBox('Results from the SVC model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults1 = QLabel('Results :')
        self.lblResults1.adjustSize()
        self.txtResults1 = QPlainTextEdit()
        self.lblAccuracy1 = QLabel('Accuracy :')
        self.txtAccuracy1 = QLineEdit()
        self.lblRoc_auc1 = QLabel('ROC_AUC :')
        self.txtRoc_auc1 = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults1)
        self.groupBox2Layout.addWidget(self.txtResults1)
        self.groupBox2Layout.addWidget(self.lblAccuracy1)
        self.groupBox2Layout.addWidget(self.txtAccuracy1)
        self.groupBox2Layout.addWidget(self.lblRoc_auc1)
        self.groupBox2Layout.addWidget(self.txtRoc_auc1)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix - SVC model
        #::--------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix :')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''

        # processing the parameters

        self.current_features = pd.DataFrame([])
        for i in range(12):
            if self.feature[i].isChecked():
                if len(self.current_features)==0:
                    self.current_features = data[features_list[i]]
                else:
                    self.current_features = pd.concat([self.current_features, data[features_list[i]]],axis=1)


        vtest_per = float(self.txtPercentTest.text())
        kernel1 = self.txtKernelType.text()
        # Clear the graphs to populate them with the new information

        self.ax1.clear()

        self.txtResults1.clear()
        self.txtResults1.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        class_le1 = LabelEncoder()
        features_list1 = self.current_features.loc[:, self.current_features.dtypes == 'object'].columns
        for i in features_list1:
            self.current_features[i]=class_le1.fit_transform(self.current_features[i])

        X = self.current_features.values

        y=data.iloc[:,-1]

        class_le = LabelEncoder()

        # fit and transform the class

        y = class_le.fit_transform(y)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vtest_per, random_state=100)

        # perform training with entropy.

        #::------------------------------------
        ##  Model 1 - gini model:

        #::------------------------------------

        #specify random forest classifier
        self.clf_svc = SVC(kernel=kernel1)

        # perform training
        self.clf_svc.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_svc.predict(X_test)
        y_pred_score = self.clf_svc.decision_function(X_test)

        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.class_rep = classification_report(y_test, y_pred)
        self.txtResults1.appendPlainText(self.class_rep)

        # accuracy score

        self.accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy1.setText(str(self.accuracy_score))

        # ROC-AUC
        self.rocauc_score = roc_auc_score(y_test,y_pred_score) * 100
        self.txtRoc_auc1.setText(str(self.rocauc_score))

        self.fpr,self.tpr, _ = roc_curve(y_test,  y_pred_score)
        self.auc = roc_auc_score(y_test, y_pred_score)

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------

        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )


        hm1 = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 15}, yticklabels=df_cm.columns, xticklabels=df_cm.columns,ax=self.ax1)

        hm1.yaxis.set_ticklabels(hm1.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        hm1.xaxis.set_ticklabels(hm1.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=10)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

    def roc_update(self):
        dialog = CanvasWindow(self)
        dialog.m.plot()
        dialog.m.ax.plot(self.fpr, self.tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % self.auc)
        dialog.m.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        dialog.m.ax.set_title('Receiver operating characteristic example')
        dialog.m.ax.set_xlim([0.0, 1.0])
        dialog.m.ax.set_ylim([0.0, 1.05])
        dialog.m.ax.set_xlabel("False Positive Rate")
        dialog.m.ax.set_ylabel("True Positive Rate")
        dialog.m.ax.legend(loc="lower right")
        dialog.m.draw()
        dialog.show()

class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)

class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvaas containing the plot for the initial analysis
    #;;----------------------------------
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Distribution'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

class Histogram_plots(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Histogram_plots, self).__init__()
        self.Title = "Histograms"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Select One of the Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature=[]

        for i in range(13):
            self.feature.append(QCheckBox(features_list_hist[i],self))

        for i in self.feature:
            i.setChecked(False)

        self.btnExecute = QPushButton("Plot")

        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature[0],0,0)
        self.groupBox1Layout.addWidget(self.feature[1],0,1)
        self.groupBox1Layout.addWidget(self.feature[2],1,0)
        self.groupBox1Layout.addWidget(self.feature[3],1,1)
        self.groupBox1Layout.addWidget(self.feature[4],2,0)
        self.groupBox1Layout.addWidget(self.feature[5],2,1)
        self.groupBox1Layout.addWidget(self.feature[6],3,0)
        self.groupBox1Layout.addWidget(self.feature[7],3,1)
        self.groupBox1Layout.addWidget(self.feature[8],4,0)
        self.groupBox1Layout.addWidget(self.feature[9],4,1)
        self.groupBox1Layout.addWidget(self.feature[10],5,0)
        self.groupBox1Layout.addWidget(self.feature[11],5,1)
        self.groupBox1Layout.addWidget(self.feature[12],6,0)
        self.groupBox1Layout.addWidget(self.btnExecute,7,1)

        self.fig1,self.ax1 = plt.subplots()
        self.axes=[self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Histogram Plot :')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1200, 900)
        self.show()
    def Message(self):
        QMessageBox.about(self , "Warning" , " You can't exceed more than 1 feature")

    def update(self):
        self.current_features = pd.DataFrame([])
        x_a=''
        work=0
        for i in range(13):
            if self.feature[i].isChecked():
                if len(self.current_features)>1:
                    #print("No")
                    self.Message()
                    work=1
                    break
                    # self.initUi()

                    #self.update()
                elif len(self.current_features)==0:
                    self.current_features = data[features_list_hist[i]]
                    x_a=features_list_hist[i]

                # else:
                #     pass

        if work == 0:
            self.ax1.clear()
            self.current_features.value_counts().plot(kind='bar',ax=self.ax1)
            self.ax1.set_title('Histogram of : '+x_a)
            self.ax1.set_xlabel(x_a)
            self.ax1.set_ylabel('frequency')
            self.fig1.tight_layout()
            self.fig1.canvas.draw_idle()

class Scatter_plots(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Scatter_plots, self).__init__()
        self.Title = "Scatter Plots"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Select X-variable here')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)
        self.work_x=0
        self.work_y=0
        # self.current_features = pd.DataFrame([])
        self.feature1=[]

        for i in range(13):
            self.feature1.append(QCheckBox(features_list_hist[i],self))

        for i in self.feature1:
            i.setChecked(False)

        self.btnSelectx = QPushButton("Select X-variable")
        self.btnSelectx.clicked.connect(self.select_x)

        self.groupBox1Layout.addWidget(self.feature1[0],0,0)
        self.groupBox1Layout.addWidget(self.feature1[1],0,1)
        self.groupBox1Layout.addWidget(self.feature1[2],0,2)
        self.groupBox1Layout.addWidget(self.feature1[3],0,3)
        self.groupBox1Layout.addWidget(self.feature1[4],0,4)
        self.groupBox1Layout.addWidget(self.feature1[5],0,5)
        self.groupBox1Layout.addWidget(self.feature1[6],0,6)
        self.groupBox1Layout.addWidget(self.feature1[7],1,0)
        self.groupBox1Layout.addWidget(self.feature1[8],1,1)
        self.groupBox1Layout.addWidget(self.feature1[9],1,2)
        self.groupBox1Layout.addWidget(self.feature1[10],1,3)
        self.groupBox1Layout.addWidget(self.feature1[11],1,4)
        self.groupBox1Layout.addWidget(self.feature1[12],1,5)
        self.groupBox1Layout.addWidget(self.btnSelectx,2,0)

        self.groupBox2 = QGroupBox('Select Y-variable here')
        self.groupBox2Layout= QGridLayout()   # Grid
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.feature2=[]

        for i in range(13):
            self.feature2.append(QCheckBox(features_list_hist[i],self))

        for i in self.feature2:
            i.setChecked(False)

        self.btnSelecty = QPushButton("Select Y-variable")
        self.btnSelecty.clicked.connect(self.select_y)

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox2Layout.addWidget(self.feature2[0],0,0)
        self.groupBox2Layout.addWidget(self.feature2[1],0,1)
        self.groupBox2Layout.addWidget(self.feature2[2],1,0)
        self.groupBox2Layout.addWidget(self.feature2[3],1,1)
        self.groupBox2Layout.addWidget(self.feature2[4],2,0)
        self.groupBox2Layout.addWidget(self.feature2[5],2,1)
        self.groupBox2Layout.addWidget(self.feature2[6],3,0)
        self.groupBox2Layout.addWidget(self.feature2[7],3,1)
        self.groupBox2Layout.addWidget(self.feature2[8],4,0)
        self.groupBox2Layout.addWidget(self.feature2[9],4,1)
        self.groupBox2Layout.addWidget(self.feature2[10],5,0)
        self.groupBox2Layout.addWidget(self.feature2[11],5,1)
        self.groupBox2Layout.addWidget(self.feature2[12],6,0)
        self.groupBox2Layout.addWidget(self.btnSelecty,6,1)
        self.groupBox2Layout.addWidget(self.btnExecute,7,1)

        self.fig1,self.ax1 = plt.subplots()
        self.axes=[self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Scatter Plot :')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        self.layout.addWidget(self.groupBox1,0,1)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG1,1,1)

        self.setCentralWidget(self.main_widget)
        self.resize(1200, 900)
        self.show()
    def Message_x(self):
        QMessageBox.about(self , "Warning" , " You selected more than 1 X-variable")
    def Message_y(self):
        QMessageBox.about(self , "Warning" , " You selected more than 1 Y-variable")
    def select_x(self):
        self.current_x = pd.DataFrame([])
        for i in range(13):
            if self.feature1[i].isChecked():
                if len(self.current_x)>1:
                    self.work_x=1
                    self.Message_x()
                    break
                elif len(self.current_x)==0:
                    self.current_x = data[features_list_hist[i]]
                    self.x_a=features_list_hist[i]
                    self.work_x=0
        # for i in range(13):
        #     if self.feature1[i].isChecked():
        #         if len(self.current_x)==0:
        #             self.current_x = data[features_list_hist[i]]
        #             self.x_a=features_list_hist[i]
        #         else:
        #             pass

    def select_y(self):
        self.current_y = pd.DataFrame([])
        for i in range(13):
            if self.feature2[i].isChecked():
                if len(self.current_y)>1:
                    self.work_y=1
                    self.Message_y()
                    break

                elif len(self.current_y)==0:
                    self.current_y = data[features_list_hist[i]]
                    self.y_a=features_list_hist[i]
                    self.work_y=0
        # for i in range(13):
        #     if self.feature2[i].isChecked():
        #         if len(self.current_y)==0:
        #             self.current_y = data[features_list_hist[i]]
        #             self.y_a=features_list_hist[i]
        #         else:
        #             pass


    def update(self):
        if self.work_x==0 and self.work_y==0:
            self.ax1.clear()
            self.ax1.scatter(self.current_x,self.current_y)
            self.ax1.set_title('Scatter plot : '+self.y_a+' vs '+self.x_a)
            self.ax1.set_xlabel(self.x_a)
            self.ax1.set_ylabel(self.y_a)
            self.fig1.tight_layout()
            self.fig1.canvas.draw_idle()
        elif self.work_x==1 and self.work_y==0:
            self.Message_x()
        elif self.work_x==0 and self.work_y==1:
            self.Message_y()
        elif self.work_x==1 and self.work_y==1:
            self.Message_x()
            self.Message_y()

class Data_find(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Data_find, self).__init__()
        self.Title = "Select Dataset"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        # Add a group to upload dataset
        self.groupBox1 = QGroupBox('Upload the dataset')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)


        self.lblPath = QLabel('Paste your datset path :')
        self.lblPath.adjustSize()

        self.txtPath = QLineEdit(self)
        self.txtPath.setText("pranay.csv")

        self.btnUpload = QPushButton("Upload")
        self.btnUpload.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.lblPath,0,0)
        self.groupBox1Layout.addWidget(self.txtPath,0,1)
        self.groupBox1Layout.addWidget(self.btnUpload,1,0)

        self.groupBox2 = QGroupBox('List of Features')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults1 = QLabel('Features :')
        self.lblResults1.adjustSize()
        self.txtResults1 = QPlainTextEdit()

        self.groupBox2Layout.addWidget(self.lblResults1)
        self.groupBox2Layout.addWidget(self.txtResults1)

        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()
    def Message(self):
        QMessageBox.about(self , "Warning" , "given path does not exist")
    def update(self):

        path1 = self.txtPath.text()
        if os.path.isfile(path1):
            global data
            global features_list
            global class_names
            global features_list_hist
            data = pd.read_csv(path1)
            data.drop(["enrollee_id"], axis=1, inplace=True)
            data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
            features_list = data.iloc[:,:-1].columns
            features_list_hist = data.columns
            class_names = ['nojob change','job change']
            self.list1='These are the list of features in the dataset :'
            for name in data.columns:
                self.list1+='\n'+name+'\n'

            self.txtResults1.appendPlainText(self.list1)

        else:
            self.Message()


class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'The HR decision prediction'
        self.width = 800
        self.height = 500
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet("color: white;"
                               "background-color: black;"
                               "selection-color: black;"
                               "selection-background-color: white;")

        label1=QLabel(self)
        label1.setText("<font color = blue>HR Analytics Application</font>")
        label1.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        label1.move(270,50)
        label1.resize(400, 350)
        fileMenu = mainMenu.addMenu('File')
        #UploadMenu=mainMenu.addMenu(('Upload Dataset'))
        #EDAMenu = mainMenu.addMenu('EDA Analysis')
        #MLModelMenu = mainMenu.addMenu('ML Models')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.jpg'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # Upload dataset

        #::----------------------------------------
        btn = QPushButton("Upload Dataset", self)
        btn.setStyleSheet('''font-size:20px;''')
        btn.clicked.connect(self.find_dataset)
        btn.resize(150,50)
        btn.move(100,100)
        #Upload1Button = QAction(QIcon('analysis.jpg'),'Open', self)
        #Upload1Button.setStatusTip('uploading the Dataset')
        #Upload1Button.triggered.connect(self.find_dataset)
        #UploadMenu.addAction(Upload1Button)

        #::----------------------------------------
        # EDA analysis
        # create a histogram of choice
        # Create a scatterplot of choice
        #::----------------------------------------
        analysis_menu=['Histogram','Scatter Plot']

        self.comboBox=QComboBox(self)
        self.comboBox.setGeometry(500,100,200,45)
        self.comboBox.addItems(analysis_menu)
        # self.comboBox.move(200,100)

        self.btn2=QPushButton('Open',self)
        self.btn2.setGeometry(500,170,120,45)
        self.btn2.clicked.connect(self.getComboValue)
        # self.btn2.move(200,150)
		
		
        ml_menu=['Decision Tree Classifier','Random Forest Classifier','Support Vector Machine']

        self.comboBox1=QComboBox(self)
        self.comboBox1.setGeometry(300,300,200,45)
        self.comboBox1.addItems(ml_menu)
        # self.comboBox1.move(300,300)

        self.btn3=QPushButton('Open',self)
        self.btn3.setGeometry(300,370,120,45)
        self.btn3.clicked.connect(self.getComboValue1)
        # self.btn3.move(200,350)
		
        #EDA1Button = QAction(QIcon('analysis.jpg'),'histograms', self)
        #EDA1Button.setStatusTip('histograms')
        #EDA1Button.triggered.connect(self.EDA1)
        #EDAMenu.addAction(EDA1Button)

        #EDA2Button = QAction(QIcon('analysis.jpg'),'scatter plots', self)
        #EDA2Button.setStatusTip('scatter plots')
        #EDA2Button.triggered.connect(self.EDA2)
        #EDAMenu.addAction(EDA2Button)

        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        #MLModel1Button =  QAction(QIcon('model.jpg'), 'Decision Tree Entropy', self)
        #MLModel1Button.setStatusTip('ML algorithm with Entropy ')
        #MLModel1Button.triggered.connect(self.MLDT)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        #MLModel2Button = QAction(QIcon('model.jpg'), 'Random Forest Classifier', self)
        #MLModel2Button.setStatusTip('Random Forest Classifier ')
        #MLModel2Button.triggered.connect(self.MLRF)

        #::------------------------------------------------------
        # Support Vector Classifier
        #::------------------------------------------------------
        #MLModel3Button = QAction(QIcon('model.jpg'), 'Support Vector Classifier', self)
        #MLModel3Button.setStatusTip('Support Vector Classifier')
        #MLModel3Button.triggered.connect(self.MLSVM)

        #MLModelMenu.addAction(MLModel1Button)
        #MLModelMenu.addAction(MLModel2Button)
        #MLModelMenu.addAction(MLModel3Button)

        self.dialogs = list()
    def getComboValue(self):
        if self.comboBox.currentText()=='Histogram':
            self.EDA1()
        else:
            self.EDA2()
    def getComboValue1(self):
        if self.comboBox1.currentText()=='Decision Tree Classifier':
            self.MLDT()
        elif self.comboBox1.currentText()=='Random Forest Classifier':
            self.MLRF()
        else:
            self.MLSVM()
    def find_dataset(self):
        dialog = Data_find()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA1(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains the happiness.score
        # X was populated in the method data_happiness()
        # at the start of the application
        #::------------------------------------------------------
        dialog = Histogram_plots()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA2(self):
        #::------------------------------------------------------
        # Creates the histogram
        # The X variable contains the happiness.score
        # X was populated in the method data_happiness()
        # at the start of the application
        #::------------------------------------------------------
        dialog = Scatter_plots()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def MLSVM(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = SupportVector()
        self.dialogs.append(dialog)
        dialog.show()

def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    ex = App()
    ex.show()
    sys.exit(app.exec_())



# def data_hr():
#     #::--------------------------------------------------
#     # read the .csv file of HR data
#     # save column names of X- variable as a list
#     # save class_names of target 0-no job change and 1 - job change
#     #::--------------------------------------------------
#     global data
#     global features_list
#     global class_names
#     global features_list_hist
#     data = pd.read_csv('pranay.csv')
#     data.drop(["enrollee_id"], axis=1, inplace=True)
#
#     data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
#
#     features_list = data.iloc[:,:-1].columns
#     features_list_hist = data.columns
#     class_names = ['nojob change','job change']


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    # data_hr()
    main()