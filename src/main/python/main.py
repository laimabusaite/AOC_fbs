# from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import (QMainWindow, QDialog, QVBoxLayout, QPushButton, QWidget, QApplication, QLabel, QTextEdit, QFileDialog)

from PyQt5.uic import loadUiType
import sys

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

import AOC_coefficients_data_analysis as AOC
import numpy as np
import os
import re

cwd = os.getcwd()
AOC_layout = os.path.join(cwd, 'AOC_layout.ui')

Ui_MainWindow, QMainWindow = loadUiType(AOC_layout)

class AOCWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(AOCWindow, self).__init__(parent)
        self.setupUi(self)

        self.bParSingleStepDecimals = 5
        self.ymaxParSingleStepDecimals = 5
        self.bPerSingleStepDecimals = 5
        self.ymaxPerSingleStepDecimals = 5
        self.cSingleStepDecimals = 5
        self.c1SingleStepDecimals = 5

        self.aoc = AOC.AOCcoefficients()
        self.aoc.BexpName = 'Babs'
        # self.aoc.BexpName = 'B'#
        self.aoc.ParExpName = 'per_smooth'
        self.aoc.PerExpName = 'par_smooth'
        # self.aoc.ParExpName = 'diff' #'par_avg' #apzināti samainīti vietām
        # self.aoc.PerExpName = 'diff0' #'per_avg'
        # self.aoc.ParExpName = 'per_avg_mod'  # 'per_avg' #apzināti samainīti vietām
        # self.aoc.PerExpName = 'par_avg_mod'  # 'par_avg'
        # self.aoc.semPar = 'diff_sem'
        # self.aoc.semPer = 'diff0_sem'

        self.aoc.kappa = 0.0

        self.setbParSpinBox()
        self.setymaxParSpinBox()
        self.setbPerSpinBox()
        self.setymaxPerSpinBox()
        self.setcSpinBox()
        self.setc1SpinBox()
        # self.aoc.setMagneticCondition(Bmin = 50, Bmax = 2800)
        self.setMagneticRange()

        self.experimentFilename = None
        self.theoryFilenameList = None

        # self.aoc.smoothDiff = True
        # self.aoc.smoothCirc = True

        self.plotMinDiff = None
        self.plotMaxDiff = None
        self.plotMinCirc = None
        self.plotMaxCirc = None
        


        self.fig = Figure()
        self.addmpl(self.fig)

        self.selectExperimentButton.clicked.connect(self.getExpFile)
        self.displayExperimentPath.editingFinished.connect(self.getExperimentalData)

        self.selectTheoryButton.clicked.connect(self.getTheoryFiles)

        self.bestFitButton.clicked.connect(self.fitData)

        self.bParSpinBox.valueChanged.connect(self.changebParSpinBox)
        self.yMaxParSpinBox.valueChanged.connect(self.changeymaxParSpinBox)
        self.bParStep.editingFinished.connect(self.changebParStep)
        self.yMaxParStep.editingFinished.connect(self.changeymaxParStep)

        self.bPerSpinBox.valueChanged.connect(self.changebPerSpinBox)
        self.yMaxPerSpinBox.valueChanged.connect(self.changeymaxPerSpinBox)
        self.bPerStep.editingFinished.connect(self.changebPerStep)
        self.yMaxPerStep.editingFinished.connect(self.changeymaxPerStep)

        self.cSpinBox.valueChanged.connect(self.changecSpinBox)
        self.c1SpinBox.valueChanged.connect(self.changec1SpinBox)
        self.cStep.editingFinished.connect(self.changecStep)
        self.c1Step.editingFinished.connect(self.changec1Step)

        self.minB.editingFinished.connect(self.changeBmin)
        self.maxB.editingFinished.connect(self.changeBmax)

        self.fixedbParCheckBox.stateChanged.connect(self.changebParVary)
        self.fixedymaxParCheckBox.stateChanged.connect(self.changeymaxParVary)
        self.fixedbPerCheckBox.stateChanged.connect(self.changebPerVary)
        self.fixedymaxPerCheckBox.stateChanged.connect(self.changeymaxPerVary)
        self.fixedCcheckBox.stateChanged.connect(self.changecVary)
        self.fixedC1checkBox.stateChanged.connect(self.changec1Vary)

        # self.smoothDiffCheckBox.stateChanged.connect(self.changeSmoothDiff)
        # self.smoothCircCheckBox.stateChanged.connect(self.changeSmoothCirc)
        self.setButton.clicked.connect(self.pushSetButton)

        self.saveButton.clicked.connect(self.saveAll)


    def setbParSpinBox(self):
        self.bParSpinBox.setValue(self.aoc.bParManual)
        print(f'bParManual: {self.aoc.bParManual}')
        self.bParSpinBox.setDecimals(10)
        self.bParSpinBox.setSingleStep(0.1**self.bParSingleStepDecimals)
        self.bParSpinBox.setRange(-np.inf, np.inf)
        self.bParSpinBox.setKeyboardTracking(False)
        # self.bParStep.setValue(self.bParSingleStepDecimals)
        self.bParStep.clear()
        self.bParStep.insert(str(self.bParSingleStepDecimals))
    def setymaxParSpinBox(self):
        self.yMaxParSpinBox.setValue(self.aoc.ymaxParManual)
        self.yMaxParSpinBox.setDecimals(10)
        self.yMaxParSpinBox.setSingleStep(0.1**self.ymaxParSingleStepDecimals)
        self.yMaxParSpinBox.setRange(-np.inf, np.inf)
        self.yMaxParSpinBox.setKeyboardTracking(False)
        self.yMaxParStep.clear()
        self.yMaxParStep.insert(str(self.ymaxParSingleStepDecimals))

    def setbPerSpinBox(self):
        self.bPerSpinBox.setValue(self.aoc.bPerManual)
        self.bPerSpinBox.setDecimals(10)
        self.bPerSpinBox.setSingleStep(0.1**self.bPerSingleStepDecimals)
        self.bPerSpinBox.setRange(-np.inf, np.inf)
        self.bPerSpinBox.setKeyboardTracking(False)
        # self.bParStep.setValue(self.bParSingleStepDecimals)
        self.bPerStep.clear()
        self.bPerStep.insert(str(self.bParSingleStepDecimals))
    def setymaxPerSpinBox(self):
        self.yMaxPerSpinBox.setValue(self.aoc.ymaxPerManual)
        self.yMaxPerSpinBox.setDecimals(10)
        self.yMaxPerSpinBox.setSingleStep(0.1**self.ymaxPerSingleStepDecimals)
        self.yMaxPerSpinBox.setRange(-np.inf, np.inf)
        self.yMaxPerSpinBox.setKeyboardTracking(False)
        self.yMaxPerStep.clear()
        self.yMaxPerStep.insert(str(self.ymaxPerSingleStepDecimals))

    def setcSpinBox(self):
        self.cSpinBox.setValue(self.aoc.cManual)
        self.cSpinBox.setDecimals(5)
        self.cSpinBox.setSingleStep(0.1**self.cSingleStepDecimals)
        self.cSpinBox.setRange(-np.inf, np.inf)
        self.cSpinBox.setKeyboardTracking(False)
        # self.bParStep.setValue(self.bParSingleStepDecimals)
        self.cStep.clear()
        self.cStep.insert(str(self.cSingleStepDecimals))
    def setc1SpinBox(self):
        self.c1SpinBox.setValue(self.aoc.c1Manual)
        self.c1SpinBox.setDecimals(5)
        self.c1SpinBox.setSingleStep(0.1**self.c1SingleStepDecimals)
        self.c1SpinBox.setRange(-np.inf, np.inf)
        self.c1SpinBox.setKeyboardTracking(False)
        # self.bParStep.setValue(self.bParSingleStepDecimals)
        self.c1Step.clear()
        self.c1Step.insert(str(self.c1SingleStepDecimals))

    def setMagneticRange(self):
        self.minB.insert(str(self.aoc.Bmin))
        self.maxB.insert(str(self.aoc.Bmax))



    def changeymaxParSpinBox(self):
        print('changeymaxParSpinBox')      
        # self.aoc.setYParManual(self.yMaxParSpinBox.value())
        self.aoc.ymaxParManual = self.yMaxParSpinBox.value()
        self.drawPlots(self.fig)
    def changebParSpinBox(self):
        print('changebParSpinBox')
        # self.aoc.setBParManual(self.bParSpinBox.value())
        self.aoc.bParManual = self.bParSpinBox.value()
        print(f'bParManual: {self.aoc.bParManual}')
        self.drawPlots(self.fig)
    def changebParStep(self):
        self.bParSingleStepDecimals = int(self.bParStep.text())
        self.bParSpinBox.setSingleStep(0.1**self.bParSingleStepDecimals)
    def changeymaxParStep(self):
        self.ymaxParSingleStepDecimals = int(self.yMaxParStep.text())
        self.yMaxParSpinBox.setSingleStep(0.1**self.ymaxParSingleStepDecimals)

    def changeymaxPerSpinBox(self):
        print('changeymaxPerSpinBox')      
        # self.aoc.setYParManual(self.yMaxParSpinBox.value())
        self.aoc.ymaxPerManual = self.yMaxPerSpinBox.value()
        self.drawPlots(self.fig)
    def changebPerSpinBox(self):
        print('changebPerSpinBox')
        # self.aoc.setBParManual(self.bParSpinBox.value())
        self.aoc.bPerManual = self.bPerSpinBox.value()
        self.drawPlots(self.fig)
    def changebPerStep(self):
        self.bPerSingleStepDecimals = int(self.bPerStep.text())
        self.bPerSpinBox.setSingleStep(0.1**self.bPerSingleStepDecimals)
    def changeymaxPerStep(self):
        self.ymaxPerSingleStepDecimals = int(self.yMaxPerStep.text())
        self.yMaxPerSpinBox.setSingleStep(0.1**self.ymaxPerSingleStepDecimals)


    def changecSpinBox(self):
        print('changecSpinBox')      
        # self.aoc.setYParManual(self.yMaxParSpinBox.value())
        self.aoc.cManual = self.cSpinBox.value()
        self.drawPlots(self.fig)
    def changec1SpinBox(self):
        print('changec1SpinBox')
        # self.aoc.setBParManual(self.bParSpinBox.value())
        self.aoc.c1Manual = self.c1SpinBox.value()
        self.drawPlots(self.fig)
    def changecStep(self):
        self.cSingleStepDecimals = int(self.cStep.text())
        self.cSpinBox.setSingleStep(0.1**self.cSingleStepDecimals)
    def changec1Step(self):
        self.c1SingleStepDecimals = int(self.c1Step.text())
        self.c1SpinBox.setSingleStep(0.1**self.c1SingleStepDecimals)

    def changeBmin(self):
        self.aoc.Bmin = float(self.minB.text())
    def changeBmax(self):
        self.aoc.Bmax = float(self.maxB.text())

    def changebParVary(self):
        if self.fixedbParCheckBox.isChecked():
            self.aoc.bParVary = False
        else:
            self.aoc.bParVary = True

    def changeymaxParVary(self):
        if self.fixedymaxParCheckBox.isChecked():
            self.aoc.ymaxParVary = False
        else:
            self.aoc.ymaxParVary = True
    def changebPerVary(self):
        if self.fixedbPerCheckBox.isChecked():
            self.aoc.bPerVary = False
        else:
            self.aoc.bPerVary = True
    def changeymaxPerVary(self):
        if self.fixedymaxPerCheckBox.isChecked():
            self.aoc.ymaxPerVary = False
        else:
            self.aoc.ymaxPerVary = True
    def changecVary(self):
        print("changecVary")
        if self.fixedCcheckBox.isChecked():
            self.aoc.cVary = False
        else:
            self.aoc.cVary = True
        print('self.aoc.cVary', self.aoc.cVary)
    def changec1Vary(self):
        print("changec1Vary")
        if self.fixedC1checkBox.isChecked():
            self.aoc.c1Vary = False
        else:
            self.aoc.c1Vary = True
        print('self.aoc.c1Vary', self.aoc.c1Vary)

    def changeSmoothDiff(self):
        if self.smoothDiffCheckBox.isChecked():
            self.aoc.smoothDiff = True
        else:
            self.aoc.smoothDiff = False
        # self.drawPlots(self.fig)
    def changeSmoothCirc(self):
        if self.smoothCircCheckBox.isChecked():
            self.aoc.smoothCirc = True
        else:
            self.aoc.smoothCirc = False
        # self.drawPlots(self.fig)

    def changeDiffMin(self):
        try:
            self.plotMinDiff = float(self.minDiff.text())
        except:
            if self.minDiff.text() == '':
                self.plotMinDiff = None
            else:
                pass
    def changeDiffMax(self):
        try:
            self.plotMaxDiff = float(self.maxDiff.text())
        except:
            if self.maxDiff.text() == '':
                self.plotMaxDiff = None
            else:
                pass
    def changeCircMin(self):
        try:
            self.plotMinCirc = float(self.minCirc.text())
        except:
            if self.minCirc.text() == '':
                self.plotMinCirc = None
            else:
                pass
    def changeCircMax(self):
        try:
            self.plotMaxCirc = float(self.maxCirc.text())
        except:
            if self.maxCirc.text() == '':
                self.plotMaxCirc = None
            else:
                pass



    def pushSetButton(self):
        self.changeSmoothDiff()
        self.changeSmoothCirc()
        self.changeDiffMin()
        self.changeDiffMax()
        self.changeCircMin()
        self.changeCircMax()
        self.drawPlots(self.fig)




    def getExpFile(self):
        self.displayExperimentPath.clear()

        fname = QFileDialog.getOpenFileName(self, 'Open file', 
         '/home/laima/Documents/Rb-2019/Rb-experiment/',"Data files (*.txt *.dat *.csv);;All files (*.*)")
        # fname = QFileDialog.getOpenFileName(self, 'Open file', 
        #  cwd,"Data files (*.txt *.dat *.csv);;All files (*.*)")
        self.displayExperimentPath.insert(fname[0])
        self.experimentFilename = self.displayExperimentPath.text()
        self.getExperimentalData()
      

    def getExperimentalData(self):
        self.experimentFilename = self.displayExperimentPath.text()
        self.aoc.setExpFilename(self.experimentFilename)
        if self.aoc.expFilename:
            self.aoc.importExperimentalData()
            self.setbParSpinBox()
            self.setymaxParSpinBox()
            self.setbPerSpinBox()
            self.setymaxPerSpinBox()
            self.setcSpinBox()
            self.setc1SpinBox()
        self.theoryFilenameList = self.displayTheoryPath.text()

    def getTheoryFiles(self):
        self.displayTheoryPath.clear()
        fname = QFileDialog.getOpenFileNames(self, 'Open file', 
         '/home/laima/Documents/Rb-2019/D1-merged_vienadasRabi/D1-Rb85-1/gamma=0.0180-DSteps=150-lWidth=2.0',"All files (*.*)")
        # fname = QFileDialog.getOpenFileNames(self, 'Open file', 
        #  cwd,"All files (*.*)")
        self.displayTheoryPath.insert(';'.join(fname[0]))
        self.theoryFilenameList = fname[0]
        self.aoc.setTheoryFilenames(self.theoryFilenameList)
        self.aoc.importAllTheoryDataList()
        self.drawPlots(self.fig)

    def fitData(self):
        print('fitData start')
        print('self.aoc.cVary', self.aoc.c1Vary)
        print('self.aoc.c1Vary', self.aoc.c1Vary)
        self.aoc.fitAll()
        # self.drawPlots(self.fig)
        self.setbParSpinBox()
        self.setymaxParSpinBox()
        self.setbPerSpinBox()
        self.setymaxPerSpinBox()
        self.setcSpinBox()
        self.setc1SpinBox()
        print('fitData end')

    def plotExperiment(self, fig):
        print('plotExperiment')
        self.aoc.plotParallelExperiment(self.axPar)
        self.aoc.plotPerpExperiment(self.axPer)
        self.aoc.plotDiffExperiment(self.axDiff)
        self.aoc.plotCircExperiment(self.axCirc)
        fig.suptitle(self.aoc.expFilenameLabel)
            
    def plotTheory(self, fig):
        print('plotTheory')
        self.aoc.plotAllParallelTheory(self.axPar)
        self.aoc.plotAllPerpTheory(self.axPer)
        self.aoc.plotAllDiffTheory(self.axDiff)
        self.aoc.plotAllCircTheory(self.axCirc)

    def drawPlots(self, fig):
        print('drawPlots')
        self.rmmpl()
        self.addmpl(fig)
        if self.experimentFilename:
            self.plotExperiment(fig)
            
        if self.theoryFilenameList:
            self.plotTheory(fig)

        self.axDiff.set_ylim(bottom=self.plotMinDiff)
        self.axDiff.set_ylim(top=self.plotMaxDiff)
        self.axCirc.set_ylim(bottom=self.plotMinCirc)
        self.axCirc.set_ylim(top=self.plotMaxCirc)
        # self.plotMinDiff, self.plotMaxDiff = self.axDiff.get_ylim()
        # self.plotMinCirc, self.plotMaxCirc = self.axCirc.get_ylim()
        # fig.tight_layout()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas.set_window_title('test image')


    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.axPar = fig.add_subplot(221)
        self.axPer = fig.add_subplot(222)
        self.axDiff = fig.add_subplot(223)
        self.axCirc = fig.add_subplot(224)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
                self.mplWindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)  

    def rmmpl(self):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.axPar.clear()
        self.axPer.clear()
        self.axDiff.clear()
        self.axCirc.clear()
        self.fig.clear()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def saveAll(self):
        self.getSavePath()
        # self.savePath = '/home/laima/Documents/AOC fitting parameters/AOC_fbs/src/main/python/test/testTeor_exp.csv'
        self.aoc.createSaveFile()
        try:
            self.aoc.saveToCSV(self.savePath)
        except FileNotFoundError:
            pass

    def getSavePath(self):
        # getSaveFileName()
        self.rabiList = None
        # self.theoryFilenamesLabels = np.array([re.split('-',os.path.basename(theor))[2] for theor in theorfiles])
        try:
            rabiList = np.array([re.split('Rabi=',theor)[-1] for theor in self.aoc.theoryFilenamesLabels])
            print(self.aoc.theoryFilenamesLabels)
            print (rabiList)
            rabiString = '-'.join(rabiList)
            print(rabiString)
        except AttributeError:
            rabiString = None

        self.expPowerString = None
        try:
            expnamelist = re.split('_', self.aoc.expFilenameLabel)
            self.expPowerString = '-'.join(expnamelist[:4])
        except:
            self.expPowerString = None


        # savedir = 
        saveFileName = 'AOC-{}-bpar={:.5f}-ypar={:.5f}-bper={:.5f}-yper={:.5f}-c={:.5f}-c1={:.5f}-Rabi={}.csv'.format(self.expPowerString, self.aoc.bParManual, self.aoc.ymaxParManual, self.aoc.bPerManual, self.aoc.ymaxPerManual, self.aoc.cManual, self.aoc.c1Manual, rabiString)
        defaultFilename = os.path.join(cwd, saveFileName)
        fname = QFileDialog.getSaveFileName(self, 'Save file', 
         defaultFilename, "csv files (*.csv) ;; All files (*.*)")
        self.savePath = fname[0]
        # print (fname, self.savePath)

    # def saveFigure(self):
    #     pass





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AOCWindow()

    # openExp = cwd
    # openTheor = cwd

    window.show()
    exit_code = app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
