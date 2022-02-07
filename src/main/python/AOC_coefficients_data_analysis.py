#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
from scipy.signal import savgol_filter
from lmfit.models import Model
# from AOC_functions import exp_params, difference, circularity

# Component coeffiecients
def exp_params(y_exp, b, ymax):
    exp_par_norm = (y_exp-b)/(ymax-b)
    return exp_par_norm

# difference
def difference(ya, yb, c, c1):
    delta_y = (ya-yb*c)*c1
    return delta_y

#circularity
def circularity(ya, yb, c, c1=1, kappa=0):
    return (ya-yb*c)/(ya + yb*c + kappa) #*c1

class AOCcoefficients(object):
    def __init__(self):
        # self.Rabi_list = []
        # self.
        self.setRabi()
        self.setPower()
        self.setTransition()
        self.setTheoryPath()

        self.setMagneticCondition()
        # self.setExpFilename(None)

        self.BexpName = 'Babs'
        self.ParExpName = 'par_smooth'
        self.PerExpName = 'per_smooth'
        self.semPar = 'par_sem'
        self.semPer = 'per_sem'

        self.setParCoefficients()
        self.setPerCoefficients()
        self.setCManual()


        self.fittedAll = False
        self.experimentImported = False
        self.theoryImported = False
        self.axParTitle = ''
        self.axPerTitle = ''
        self.axDiffTitle = ''
        self.axCircTitle = ''

        self.componentModel = Model(exp_params)


        self.chParMin = 1e20
        self.chPerMin = 1e20
        self.chDiffMin = 1e20
        self.chCircMin = 1e20
        self.theoryDataList = []
        self.theoryDataList1 = []
        self.expData1 = []
        self.expData = []

        self.smoothDiff = False
        self.smoothCirc = False

        self.bParVary = True
        self.ymaxParVary = True
        self.bPerVary = True
        self.ymaxPerVary = True
        self.cVary = True
        self.c1Vary = True
        self.varyProc = 0.005

        self.theoryParLabelBest = ''
        self.theoryPerLabelBest = ''
        self.theoryDiffLabelBest = ''
        self.theoryCircLabelBest = ''

        self.kappa = 0.0

        # self.minDiff = None
        # self.maxDiff = None
        # self.minCirc = None
        # self.maxCirc = None

    def clearChiSquare(self):
        self.chParMin = 1e20
        self.chPerMin = 1e20
        self.chDiffMin = 1e20
        self.chCircMin = 1e20


    def setRabi(self, rabi_list = []):
        self.Rabi_list = rabi_list

    def setPower(self, power = 100):
        self.power = power

    def setTransition(self, transition = [2, 2]):
        self.transition = transition

    def setTheoryPath(self, path = '.'):
        self.theoryPath = path

    def theory_filename(self, theory_path, Rabi):
        # filename = 'Rb85-D2-Rabi={:.2f}-shift=0.0-Ge=0'.format(Rabi)
        # filename = 'Rb85-D1-Rabi={:.2f}-shift=0.0-Ge=0Rb87-D1-Rabi={:.2f}-shift=-1919.0-Ge=0'.format(Rabi, 2.16*Rabi)
        filename = 'Rb85-D1-Rabi={:.2f}-shift=0.0-Ge=0'.format(Rabi)
        return os.path.join(theory_path, filename)

    # def setExpPath(self, path = '.'):
    #   self.exp_path = '/home/laima/Documents/Rb-2019/Rb-experiment/{}-{}/exp-lif-data/'.format(pareja[0], pareja[1])

    def setExpFilename(self, expfile):
        self.expFilename = expfile
        self.expFilenameLabel = os.path.basename(expfile)
        # if expfile:
        #     self.logfile = '{}_log.txt'.format(self.expFilename[:-4])

    def setTheoryFilenames(self, theorfiles):
        self.theoryFilenamesPaths = theorfiles
        self.theoryFilenamesLabels = np.array([re.split('-',os.path.basename(theor))[2] for theor in theorfiles])

    def setSavePath(self, savepath):
        self.savePath = savepath
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

    def setParCoefficients(self, b = 0, ymax = 1.0):
        self.bParManual = b
        self.ymaxParManual = ymax
        self.fittedAll = False

    def setPerCoefficients(self, b = 0, ymax = 1.0):
        self.bPerManual = b
        self.ymaxPerManual = ymax
        self.fittedAll = False

    def setCManual(self, c = 1.0, c1 = 1.0):
        self.cManual = c
        self.c1Manual = c1
        self.fittedAll = False

    def setMagneticCondition(self, Bmin = -np.inf, Bmax = np.inf):
        self.Bmin = Bmin
        self.Bmax = Bmax


    def importExperimentalData(self):
        self.experiment_colums = [self.BexpName, self.ParExpName, self.PerExpName, self.semPar, self.semPer]
        self.expData1 = pd.read_csv(self.expFilename, header=0, usecols=self.experiment_colums).dropna()
        condexp = (self.expData1[self.BexpName] > self.Bmin) & (self.expData1[self.BexpName] < self.Bmax)
        self.expData = self.expData1[condexp]
        print('importExperimentalData')
        print (min(self.expData[self.ParExpName]), max(self.expData[self.ParExpName]))
        self.bParMin = None #min(self.expData[self.ParExpName])
        self.bParMax = None #min(self.expData[self.ParExpName])
        # if min(self.expData[self.ParExpName]) <= 0:
        #     self.bParMax = min(self.expData[self.ParExpName])
        # if min(self.expData[self.ParExpName]) > 0:
        #     self.bParMax = min(self.expData[self.ParExpName])
        # self.bParMax = min(self.expData[self.ParExpName])
        # self.setParCoefficients(b=min(self.expData[self.ParExpName]), ymax=max(self.expData[self.ParExpName]))
        self.setParCoefficients(b=max(self.expData[self.ParExpName]), ymax=min(self.expData[self.ParExpName]))
        print (self.bParManual, self.ymaxParManual)
        self.bPerMin = None #min(self.expData[self.PerExpName])
        self.bPerMax = None #min(self.expData[self.PerExpName])
        # if min(self.expData[self.PerExpName]) <= 0:
        #     self.bPerMax = min(self.expData[self.PerExpName])
        # if min(self.expData[self.PerExpName]) > 0:
        #     self.bPerMin = min(self.expData[self.PerExpName])
        # self.bPerMax = min(self.expData[self.PerExpName])
        print (self.bPerManual, self.ymaxPerManual)
        # self.setPerCoefficients(b=min(self.expData[self.PerExpName]), ymax=max(self.expData[self.PerExpName]))
        self.setPerCoefficients(b=max(self.expData[self.PerExpName]), ymax=min(self.expData[self.PerExpName]))
        print (self.bPerManual, self.ymaxPerManual)
        self.experimentImported = True
        self.fittedAll = False

    def importTheoryData(self, theoryFilenamePath):
        theor = pd.read_csv(theoryFilenamePath, header=None, delimiter=' ', names=['B', 'LIF', 'sigma+', 'sigma-', 'difference'])
        max_par = max(theor['sigma+'])
        max_per = max(theor['sigma-'])
        print (max_par, max_per)
        theor['par_norm'] = (theor['sigma+'])/max_par
        theor['per_norm'] = (theor['sigma-'])/max_per
        theor['difference_norm'] = theor['sigma+']/max_par-theor['sigma-']/max_par
        theor['circularity'] = (theor['sigma+']/max_par-theor['sigma-']/max_par)/(theor['sigma+']/max_par+theor['sigma-']/max_par)

        theor['par_norm'] = (theor['sigma+'])/max_par + self.kappa
        theor['per_norm'] = (theor['sigma-'])/max_per + self.kappa
        theor['difference_norm']=(theor['sigma+'] - theor['sigma-'])/max_par
        theor['circularity']=(theor['sigma+'] - theor['sigma-'])/(theor['sigma+'] + theor['sigma-'] + 2*self.kappa*max_par)

        self.theorData1 = theor
        condtheor = (self.theorData1['B'] > self.Bmin) & (self.theorData1['B'] < self.Bmax)
        self.theorData = self.theorData1[condtheor]

    def importAllTheoryDataList(self):
        theor_list = []
        theor_list1 = []
        for theorfile in self.theoryFilenamesPaths:
            self.importTheoryData(theorfile)
            theor_list1.append(self.theorData1)
            theor_list.append(self.theorData)
        self.theoryDataList = theor_list
        self.theoryDataList1 = theor_list1
        self.theoryImported = True

############################################################################################################################
    def fitParallelComponent(self):
        print('fitParallelComponent')
        interpolation_function_par = interp1d(self.theorData['B'], self.theorData['par_norm'], fill_value="extrapolate")
        theor_par_interp = interpolation_function_par(self.expData[self.BexpName])
        print(f'b = {self.bParManual}, ymax = {self.ymaxParManual}')
        params = self.componentModel.make_params(b = self.bParManual, ymax = self.ymaxParManual)
        params.update(self.componentModel.make_params())
        print(f'b = {self.bParManual}, ymax = {self.ymaxParManual}')
        params['b'].set(value = self.bParManual, min=self.bParMin, max=self.bParMax, vary=self.bParVary)
        params['ymax'].set(value = self.ymaxParManual, vary=self.ymaxParVary)
        print(params)
        self.fittedExpPar = self.componentModel.fit(theor_par_interp, params, y_exp=self.expData[self.ParExpName])
        # self.fittedExpPar1 = exp_params(self.expData1[self.ParExpName], self.fittedExpPar.params['b'].value, self.fittedExpPar.params['ymax'].value)
        self.chPar = self.fittedExpPar.chisqr
        print(self.fittedExpPar.fit_report())
        print(stats.chisquare(self.fittedExpPar.best_fit, theor_par_interp, axis=None))
        print(self.fittedExpPar.redchi)
        print(np.sum((np.array(self.fittedExpPar.best_fit)-np.array(theor_par_interp))**2))
        stats_ch2 = stats.chisquare(theor_par_interp, self.fittedExpPar.best_fit)
        stats_ch22 = stats.chisquare(self.fittedExpPar.best_fit, theor_par_interp)
        print('self.fittedExpPar.chisqr', self.chPar, 'stats_ch2', stats_ch2, 'stats_ch22', stats_ch22)
        if self.chPar < self.chParMin:
            self.theoryParLabelBest = self.theoryFilenameLabel
            self.bParBest = self.fittedExpPar.params['b'].value
            self.ymaxParBest = self.fittedExpPar.params['ymax'].value
            self.expDataParBest = self.fittedExpPar.best_fit
            # self.expDataParBest1 = self.fittedExpPar1
            self.chParMin = self.chPar
            self.fitReportPar = self.fittedExpPar.fit_report()
            self.axParTitle = 'Best: {}, b = {:.5f}, y_max = {:.5f}, chisqr={:.3f}'.format(self.theoryParLabelBest, self.bParBest, self.ymaxParBest, self.chParMin)
        # self.fittedAll = True

    def fitPerpComponent(self):
        interpolation_function_per = interp1d(self.theorData['B'], self.theorData['per_norm'], fill_value="extrapolate")
        theor_per_interp = interpolation_function_per(self.expData[self.BexpName])
        params = self.componentModel.make_params(b = self.bPerManual, ymax = self.ymaxPerManual)
        params['b'].set(min=self.bPerMin, max=self.bPerMax, vary=self.bPerVary)
        params['ymax'].set(vary=self.ymaxPerVary)
        self.fittedExpPer = self.componentModel.fit(theor_per_interp, params, y_exp=self.expData[self.PerExpName])
        # self.fittedExpPer1 = exp_params(self.expData1[self.PerExpName], self.fittedExpPer.params['b'].value, self.fittedExpPer.params['ymax'].value)
        self.chPer = self.fittedExpPer.chisqr
        print (self.fittedExpPer.fit_report())
        print (stats.chisquare(self.fittedExpPer.best_fit, theor_per_interp, axis=None))
        print (self.fittedExpPer.redchi)
        print (np.sum((np.array(self.fittedExpPer.best_fit)-np.array(theor_per_interp))**2))
        if self.chPer < self.chPerMin:
            self.theoryPerLabelBest = self.theoryFilenameLabel
            self.bPerBest = self.fittedExpPer.params['b'].value
            self.ymaxPerBest = self.fittedExpPer.params['ymax'].value
            self.expDataPerBest = self.fittedExpPer.best_fit
            # self.expDataPerBest1 = self.fittedExpPer1
            self.chPerMin = self.chPer
            self.fitReportPer = self.fittedExpPer.fit_report()
            self.axPerTitle = 'Best: {}, b = {:.5f}, y_max = {:.5f}, chisqr={:.3f}'.format(self.theoryPerLabelBest, self.bPerBest, self.ymaxPerBest, self.chPerMin)


    def difference_ya(self, ya, c, c1):
        delta_y = difference(ya, self.fittedExpPer.best_fit, c, c1)
        delta_y = savgol_filter(delta_y, 401, 2)
        return delta_y

    def fitDifference(self):
        self.difference_model = Model(self.difference_ya)
        interpolation_function_diff = interp1d(self.theorData['B'], self.theorData['difference_norm'], fill_value="extrapolate")
        theor_diff_interp = interpolation_function_diff(self.expData[self.BexpName])
        cinit = self.cManual
        cmin = None #cinit - 0.1*np.abs(cinit)
        cmax = None #cinit + 0.1*np.abs(cinit)
        c1init = self.c1Manual
        c1min = None #c1init - 0.1*np.abs(c1init)
        c1max = None #c1init + 0.1*np.abs(c1init)
        params = self.difference_model.make_params(c=self.cManual, c1=self.c1Manual)
        params['c'].set(min=cmin, max=cmax, vary=self.cVary)
        params['c1'].set(min=c1min, max=c1max, vary=self.c1Vary)
        self.fittedExpDiff = self.difference_model.fit(theor_diff_interp, params, ya=self.fittedExpPar.best_fit)
        self.chDiff = self.fittedExpDiff.chisqr
        stats_ch2 = stats.chisquare(theor_diff_interp, self.fittedExpDiff.best_fit)
        stats_ch22 = stats.chisquare(self.fittedExpDiff.best_fit, theor_diff_interp)
        print('self.fittedExpDiff.chisqr', self.fittedExpDiff.chisqr, 'stats_ch2', stats_ch2, 'stats_ch22', stats_ch22)
        if self.chDiff < self.chDiffMin:
            self.theoryDiffLabelBest = self.theoryFilenameLabel
            self.cDiffBest = self.fittedExpDiff.params['c'].value
            self.c1DiffBest = self.fittedExpDiff.params['c1'].value
            self.expDataDiffBest = self.fittedExpDiff.best_fit
            # self.expDataDiffBest1 = self.fittedExpDiff1
            self.chDiffMin = self.chDiff
            self.fitReportDiff = self.fittedExpDiff.fit_report()
            self.axDiffTitle = 'Best: {}, c = {:.5f}, c1 = {:.5f}, chisqr={:.3f}'.format(self.theoryDiffLabelBest, self.cDiffBest, self.c1DiffBest, self.chDiffMin)

    def difference_4(self, ya, b, ymax, c, c1):
        y = exp_params(ya, b, ymax)
        delta_y = difference(y, self.fittedExpPer.best_fit, c, c1)
        delta_y = savgol_filter(delta_y, 401, 2)
        return delta_y

    def fitDifference4(self):
        print ('fitDifference4')
        print(self.bParVary,
            self.ymaxParVary,
            self.bPerVary,
            self.ymaxPerVary,
            self.cVary,
            self.c1Vary)
        self.difference_model4 = Model(self.difference_4)
        interpolation_function_diff = interp1d(self.theorData['B'], self.theorData['difference_norm'], fill_value="extrapolate")
        theor_diff_interp = interpolation_function_diff(self.expData[self.BexpName])
        binit = self.fittedExpPar.params['b'].value
        bmin = binit - self.varyProc*np.abs(binit)
        bmax = binit + 0.1*np.abs(binit) #min(self.bParMax, binit + self.varyProc*np.abs(binit)) #
        ymaxinit = self.fittedExpPar.params['ymax'].value
        ymaxmin = ymaxinit - self.varyProc*np.abs(ymaxinit)
        ymaxmax = ymaxinit + self.varyProc*np.abs(ymaxinit)
        cinit = self.cManual
        cmin = None #cinit - 0.1*np.abs(cinit)
        cmax = None #cinit + 0.1*np.abs(cinit)
        c1init = self.c1Manual
        c1min = None #c1init - 0.1*np.abs(c1init)
        c1max = None #c1init + 0.1*np.abs(c1init)
        params = self.difference_model4.make_params(b = binit , ymax = ymaxinit, c=self.cManual, c1=self.c1Manual)
        params['b'].set(min=bmin, max=bmax, vary=self.bParVary)
        params['ymax'].set(min=ymaxmin, max=ymaxmax, vary=self.ymaxParVary)
        params['c'].set(min=cmin, max=cmax, vary=self.cVary)
        params['c1'].set(min=c1min, max=c1max, vary=self.c1Vary)
        self.fittedExpDiff = self.difference_model4.fit(theor_diff_interp, params, ya=self.expData[self.ParExpName])
        self.chDiff = self.fittedExpDiff.chisqr
        stats_ch2 = stats.chisquare(theor_diff_interp, self.fittedExpDiff.best_fit)
        stats_ch22 = stats.chisquare(self.fittedExpDiff.best_fit, theor_diff_interp)
        print('self.fittedExpDiff4.chisqr', self.fittedExpDiff.chisqr, 'stats_ch2', stats_ch2, 'stats_ch22', stats_ch22)
        if self.chDiff < self.chDiffMin:
            self.theoryDiffLabelBest = self.theoryFilenameLabel
            self.bParBest = self.fittedExpDiff.params['b'].value
            self.ymaxParBest = self.fittedExpDiff.params['ymax'].value
            self.cDiffBest = self.fittedExpDiff.params['c'].value
            self.c1DiffBest = self.fittedExpDiff.params['c1'].value
            self.expDataDiffBest = self.fittedExpDiff.best_fit
            # self.expDataDiffBest1 = self.fittedExpDiff1
            self.chDiffMin = self.chDiff
            self.fitReportDiff = self.fittedExpDiff.fit_report()
            self.axDiffTitle = 'Best: {}, b = {:.5f}, ymax = {:.5f}, c = {:.5f}, c1 = {:.5f}, chisqr={:.3f}'.format(self.theoryDiffLabelBest, self.bParBest, self.ymaxParBest, self.cDiffBest, self.c1DiffBest, self.chDiffMin)



    def difference_6(self, ya, b1, ymax1, b2, ymax2, c, c1):
        y1 = exp_params(ya, b1, ymax1)
        y2 = exp_params(self.expData[self.PerExpName], b2, ymax2)
        delta_y = difference(y1, y2, c, c1)
        delta_y = savgol_filter(delta_y, 401, 2)
        return delta_y

    def fitDifference6(self):
        print ('fitDifference6')
        print(self.bParVary,
            self.ymaxParVary,
            self.bPerVary,
            self.ymaxPerVary,
            self.cVary,
            self.c1Vary)
        self.difference_model6 = Model(self.difference_6)
        interpolation_function_diff = interp1d(self.theorData['B'], self.theorData['difference_norm'], fill_value="extrapolate")
        theor_diff_interp = interpolation_function_diff(self.expData[self.BexpName])
        b1init = self.fittedExpPar.params['b'].value #self.bParManual #self.fittedExpPar.params['b'].value
        b1min = b1init - self.varyProc*np.abs(b1init)
        b1max = b1init + 0.1*np.abs(b1init) #min(self.bParMax, b1init + self.varyProc*np.abs(b1init)) #b1init + 0.1*np.abs(b1init)
        b2init = self.fittedExpPer.params['b'].value #self.bPerManual #self.fittedExpPar.params['b'].value
        b2min = b2init - self.varyProc*np.abs(b2init)
        b2max = b2init + 0.1*np.abs(b2init) #min(self.bPerMax, b2init + self.varyProc*np.abs(b2init)) #b2init + 0.1*np.abs(b2init)
        ymax1init = self.fittedExpPar.params['ymax'].value #self.ymaxParManual #self.fittedExpPar.params['ymax'].value
        ymax1min = ymax1init - self.varyProc*np.abs(ymax1init)
        ymax1max = ymax1init + self.varyProc*np.abs(ymax1init)
        ymax2init = self.fittedExpPer.params['ymax'].value #self.ymaxPerManual #self.fittedExpPar.params['ymax'].value
        ymax2min = ymax2init - self.varyProc*np.abs(ymax2init)
        ymax2max = ymax2init + self.varyProc*np.abs(ymax2init)
        cinit = self.cManual
        cmin = None #cinit - 0.1*np.abs(cinit)
        cmax = None #cinit + 0.1*np.abs(cinit)
        c1init = self.c1Manual
        c1min = None #c1init - 0.1*np.abs(c1init)
        c1max = None #c1init + 0.1*np.abs(c1init)
        print('init', cinit, cmin, cmax, c1init, c1min, c1max)
        params = self.difference_model6.make_params(b1 = b1init, ymax1 = ymax1init, b2 = b2init, ymax2 = ymax2init, c=cinit, c1=c1init)
        params['b1'].set(min=b1min, max=b1max, vary=self.bParVary)
        params['ymax1'].set(min=ymax1min, max=ymax1max, vary=self.ymaxParVary)
        params['b2'].set(min=b2min, max=b2max, vary=self.bPerVary)
        params['ymax2'].set(min=ymax2min, max=ymax2max, vary=self.ymaxPerVary)
        params['c'].set(min=cmin, max=cmax, vary=self.cVary)
        params['c1'].set(min=c1min, max=c1max, vary=self.c1Vary)
        self.fittedExpDiff = self.difference_model6.fit(theor_diff_interp, params, ya=self.expData[self.ParExpName])
        self.chDiff = self.fittedExpDiff.chisqr
        stats_ch2 = stats.chisquare(theor_diff_interp, self.fittedExpDiff.best_fit)
        stats_ch22 = stats.chisquare(self.fittedExpDiff.best_fit, theor_diff_interp)
        print('self.fittedExpDiff6.chisqr', self.fittedExpDiff.chisqr, 'stats_ch2', stats_ch2, 'stats_ch22', stats_ch22)
        if self.chDiff < self.chDiffMin:
            self.theoryDiffLabelBest = self.theoryFilenameLabel
            self.bParBest = self.fittedExpDiff.params['b1'].value
            self.ymaxParBest = self.fittedExpDiff.params['ymax1'].value
            self.bPerBest = self.fittedExpDiff.params['b2'].value
            self.ymaxPerBest = self.fittedExpDiff.params['ymax2'].value
            self.cDiffBest = self.fittedExpDiff.params['c'].value
            self.c1DiffBest = self.fittedExpDiff.params['c1'].value
            self.expDataDiffBest = self.fittedExpDiff.best_fit
            # self.expDataDiffBest1 = self.fittedExpDiff1
            self.chDiffMin = self.chDiff
            self.fitReportDiff = self.fittedExpDiff.fit_report()
            self.axDiffTitle = 'Best: {}, b = {:.5f}, ymax = {:.5f}, c = {:.5f}, c1 = {:.5f}, chisqr={:.3f}'.format(self.theoryDiffLabelBest, self.bParBest, self.ymaxParBest, self.cDiffBest, self.c1DiffBest, self.chDiffMin)


    def fitAll(self):
        print ('start fitAll')
        print(self.bParVary,
            self.ymaxParVary,
            self.bPerVary,
            self.ymaxPerVary,
            self.cVary,
            self.c1Vary)
        self.clearChiSquare()
        condtheor = (self.theorData1['B'] > self.Bmin) & (self.theorData1['B'] < self.Bmax)
        self.theorData = self.theorData1[condtheor]
        condexp = (self.expData1[self.BexpName] > self.Bmin) & (self.expData1[self.BexpName] < self.Bmax)
        self.expData = self.expData1[condexp]
        for i in range(len(self.theoryDataList)):
            self.theorData = self.theoryDataList[i]
            self.theorData1 = self.theoryDataList1[i]
            self.theoryFilenameLabel = self.theoryFilenamesLabels[i]
            self.fitParallelComponent()
            self.fitPerpComponent()
            self.fitDifference()
            # self.fitDifference4()
            # self.fitDifference6()
        self.bParManual = self.bParBest
        self.ymaxParManual = self.ymaxParBest
        self.bPerManual = self.bPerBest
        self.ymaxPerManual = self.ymaxPerBest
        self.cManual = self.cDiffBest
        self.c1Manual = self.c1DiffBest

        print('cManual:', self.cManual, 'c1Manual:', self.c1Manual)
        
        self.fittedAll = True


    def plotParallelTheory(self, ax = plt):
        thfig = ax.plot(self.theorData['B'], self.theorData['par_norm'], label = self.theoryFilenameLabel, zorder=5)
        color = thfig[0].get_color()
        if len(self.theorData) <  len(self.theorData1):
            ax.plot(self.theorData1['B'], self.theorData1['par_norm'], ls='dashed', c = color, label='', zorder=5)
        # ax.plot(self.theorData['B'], self.theorData['par_norm'], label = 'Rabi = {:.2f} MHz'.format(self.rabi))
        # ax.grid(b=True, which='both', color='0.65', linestyle='-')

    def plotPerpTheory(self, ax = plt):
        thfig = ax.plot(self.theorData['B'], self.theorData['per_norm'], label = self.theoryFilenameLabel, zorder=5)
        color = thfig[0].get_color()
        if len(self.theorData) <  len(self.theorData1):
            ax.plot(self.theorData1['B'], self.theorData1['per_norm'], ls='dashed', c = color, label='', zorder=5)
        # ax.plot(self.theorData['B'], self.theorData['par_norm'], label = 'Rabi = {:.2f} MHz'.format(self.rabi))
        # ax.grid(b=True, which='both', color='0.65', linestyle='-')

    def plotDiffTheory(self, ax = plt):
        thfig = ax.plot(self.theorData['B'], self.theorData['difference_norm'], label = self.theoryFilenameLabel, zorder=5)
        color = thfig[0].get_color()
        if len(self.theorData) <  len(self.theorData1):
            ax.plot(self.theorData1['B'], self.theorData1['difference_norm'], ls='dashed', c = color, label='', zorder=5)
        # ax.grid(b=True, which='both', color='0.65', linestyle='-')

    def plotCircTheory(self, ax = plt):
        thfig = ax.plot(self.theorData['B'], self.theorData['circularity'], label = self.theoryFilenameLabel, zorder=5)
        color = thfig[0].get_color()
        if len(self.theorData) <  len(self.theorData1):
            ax.plot(self.theorData1['B'], self.theorData1['circularity'], ls='dashed', c = color, label='', zorder=5)
        # ax.grid(b=True, which='both', color='0.65', linestyle='-')
        

    def plotAllParallelTheory(self, ax = plt):
        for i in range(len(self.theoryDataList)):
            self.theorData = self.theoryDataList[i]
            self.theorData1 = self.theoryDataList1[i]
            self.theoryFilenameLabel = self.theoryFilenamesLabels[i]
            self.plotParallelTheory(ax)
        ax.grid(b=True, which='both', color='0.65', linestyle='-')
        ax.legend()

    def plotAllPerpTheory(self, ax = plt):
        for i in range(len(self.theoryDataList)):
            self.theorData = self.theoryDataList[i]
            self.theorData1 = self.theoryDataList1[i]
            self.theoryFilenameLabel = self.theoryFilenamesLabels[i]
            self.plotPerpTheory(ax)
        ax.grid(b=True, which='both', color='0.65', linestyle='-')
        ax.legend()

    def plotAllDiffTheory(self, ax = plt):
        for i in range(len(self.theoryDataList)):
            self.theorData = self.theoryDataList[i]
            self.theorData1 = self.theoryDataList1[i]
            self.theoryFilenameLabel = self.theoryFilenamesLabels[i]
            self.plotDiffTheory(ax)
        ax.grid(b=True, which='both', color='0.65', linestyle='-')
        ax.legend()

    def plotAllCircTheory(self, ax = plt):
        for i in range(len(self.theoryDataList)):
            self.theorData = self.theoryDataList[i]
            self.theorData1 = self.theoryDataList1[i]
            self.theoryFilenameLabel = self.theoryFilenamesLabels[i]
            self.plotCircTheory(ax)
        ax.grid(b=True, which='both', color='0.65', linestyle='-')
        ax.legend()


    def manualParallelExperiment(self):
        self.axParTitle = 'Best: {}, b = {:.5f}, y_max = {:.5f}'.format(self.theoryParLabelBest, self.bParManual, self.ymaxParManual)#, self.chParMin)
        self.manualExpPar = exp_params(self.expData[self.ParExpName], self.bParManual, self.ymaxParManual)
        if len(self.expData) <  len(self.expData1):
            self.manualExpPar1 = exp_params(self.expData1[self.ParExpName], self.bParManual, self.ymaxParManual)

    def manualPerpExperiment(self):
        self.axPerTitle = 'Best: {}, b = {:.5f}, y_max = {:.5f}'.format(self.theoryPerLabelBest, self.bPerManual, self.ymaxPerManual)#, self.chParMin)
        self.manualExpPer = exp_params(self.expData[self.PerExpName], self.bPerManual, self.ymaxPerManual)
        if len(self.expData) <  len(self.expData1):
            self.manualExpPer1 = exp_params(self.expData1[self.PerExpName], self.bPerManual, self.ymaxPerManual)

    def manualDiffExperiment(self):
        self.axDiffTitle = 'Best: {}, c = {:.5f}, c1 = {:.5f}'.format(self.theoryDiffLabelBest, self.cManual, self.c1Manual)#, self.chParMin)
        self.manualExpDiff = difference(self.manualExpPar, self.manualExpPer, self.cManual, self.c1Manual)
        if self.smoothDiff:
            self.manualExpDiffSmooth = savgol_filter(self.manualExpDiff, 401, 2, mode='nearest')
        if len(self.expData) <  len(self.expData1):
            self.manualExpDiff1 = difference(self.manualExpPar1, self.manualExpPer1, self.cManual, self.c1Manual)
            if self.smoothDiff:
                self.manualExpDiffSmooth1 = savgol_filter(self.manualExpDiff1, 401, 2, mode='nearest')

    def manualCircExperiment(self):
        self.axCircTitle = 'Best: {}, c = {:.5f}, c1 = {:.5f}'.format(self.theoryCircLabelBest, self.cManual, self.c1Manual)#, self.chParMin)
        self.manualExpCirc = circularity(self.manualExpPar, self.manualExpPer, self.cManual, self.c1Manual, kappa=self.kappa)
        if self.smoothCirc:
            self.manualExpCircSmooth = savgol_filter(self.manualExpCirc, 401, 2, mode='nearest')
        if len(self.expData) <  len(self.expData1):
            self.manualExpCirc1 = circularity(self.manualExpPar1, self.manualExpPer1, self.cManual, self.c1Manual, kappa=self.kappa)
            if self.smoothCirc:
                self.manualExpCircSmooth1 = savgol_filter(self.manualExpCirc1, 401, 2, mode='nearest')




    def plotParallelExperiment(self, ax = plt):
        self.manualParallelExperiment()
        ax.scatter(self.expData[self.BexpName], self.manualExpPar, s=1, c='k', label='', zorder=3)
        if len(self.expData) <  len(self.expData1):
                ax.scatter(self.expData1[self.BexpName], self.manualExpPar1, s=1, c='grey',  zorder=1, label='') 
        try:
            ax.set_title(self.axParTitle, fontsize='medium')
        except AttributeError:
            ax.title(self.axParTitle, fontsize='medium')
        ax.grid(b=True, which='both', color='0.65', linestyle='-')


    def plotPerpExperiment(self, ax = plt):
        self.manualPerpExperiment()
        ax.scatter(self.expData[self.BexpName], self.manualExpPer, s=1, c='k', label='', zorder=3)
        if len(self.expData) <  len(self.expData1):
                ax.scatter(self.expData1[self.BexpName], self.manualExpPer1, s=1, c='grey',  zorder=1, label='')
        try:
            ax.set_title(self.axPerTitle, fontsize='medium')
        except AttributeError:
            ax.title(self.axPerTitle, fontsize='medium')
        ax.grid(b=True, which='both', color='0.65', linestyle='-')

    def plotDiffExperiment(self, ax =plt):
        self.manualParallelExperiment()
        self.manualPerpExperiment()
        self.manualDiffExperiment()
        if self.smoothDiff:
            if len(self.expData) <  len(self.expData1):
                ax.scatter(self.expData1[self.BexpName], self.manualExpDiffSmooth1, s=1, c='grey',  zorder=3, label='')
            ax.scatter(self.expData[self.BexpName], self.manualExpDiffSmooth, s=1, c='k', label='', zorder=4)
        else:
            if len(self.expData) <  len(self.expData1):
                ax.scatter(self.expData1[self.BexpName], self.manualExpDiff1, s=1, c='grey',  zorder=1, label='')
            ax.scatter(self.expData[self.BexpName], self.manualExpDiff, s=1, c='k', label='', zorder=2)
        try:
            ax.set_title(self.axDiffTitle, fontsize='medium')
        except AttributeError:
            ax.title(self.axDiffTitle, fontsize='medium')
        ax.grid(b=True, which='both', color='0.65', linestyle='-')

    def plotCircExperiment(self, ax =plt):
        self.manualCircExperiment()
        if self.smoothCirc:
            if len(self.expData) <  len(self.expData1):
                ax.scatter(self.expData1[self.BexpName], self.manualExpCircSmooth1, s=1, c='grey',  zorder=3, label='')
            ax.scatter(self.expData[self.BexpName], self.manualExpCircSmooth, s=1, c='k', label='', zorder=4)
        else:
            if len(self.expData) <  len(self.expData1):
                ax.scatter(self.expData1[self.BexpName], self.manualExpCirc1, s=1, c='grey',  zorder=1, label='')
            ax.scatter(self.expData[self.BexpName], self.manualExpCirc, s=1, c='k', label='', zorder=2)
        try:
            ax.set_title(self.axCircTitle, fontsize='medium')
        except AttributeError:
            ax.title(self.axCircTitle, fontsize='medium')
        ax.grid(b=True, which='both', color='0.65', linestyle='-')


    def createSaveExp(self):
        if len(self.expData) <  len(self.expData1):
            savedf = pd.DataFrame(self.expData1[self.BexpName])
            # savedf.rename(columns={self.BexpName: 'B'}, inplace=True)
            savedf.rename(columns={self.BexpName: 'Magnetic field'}, inplace=True)
            savedf['Parallel component'] = np.array(self.manualExpPar1)
            savedf['Parallel SEM'] = self.expData1[self.semPar] / (self.ymaxParManual - self.bParManual)
            savedf['Perpendicular component'] = np.array(self.manualExpPer1)
            savedf['Perpendicular SEM'] = self.expData1[self.semPer] / (self.ymaxPerManual - self.bPerManual)
            savedf['Difference'] = np.array(self.manualExpDiff1)
            savedf['Circularity'] = np.array(self.manualExpCirc1)
            if self.smoothDiff:
                savedf['Difference smooth'] = np.array(self.manualExpDiffSmooth1)
            savedf['Circularity smooth'] = np.array(self.manualExpCirc1)
            if self.smoothCirc:
                savedf['Difference smooth'] = np.array(self.manualExpCircSmooth1)
            # self.saveExpDataframe = savedf
            
        else:
            # savedf = self.expData[self.BexpName]
            savedf = pd.DataFrame(self.expData[self.BexpName])
            # savedf.rename(columns={self.BexpName: 'B'}, inplace=True)
            savedf.rename(columns={self.BexpName: 'Magnetic field'}, inplace=True)
            savedf['Parallel component'] = np.array(self.manualExpPar)
            savedf['Parallel SEM'] = self.expData[self.semPar] / (self.ymaxParManual - self.bParManual)
            savedf['Perpendicular component'] = np.array(self.manualExpPer)
            savedf['Perpendicular SEM'] = self.expData[self.semPer] / (self.ymaxPerManual - self.bPerManual)
            savedf['Difference'] = np.array(self.manualExpDiff)
            if self.smoothDiff:
                savedf['Difference smooth'] = np.array(self.manualExpDiffSmooth)
            else:
                savedf['Difference smooth'] = ''
            savedf['Circularity'] = np.array(self.manualExpCirc)
            if self.smoothCirc:
                savedf['Circularity smooth'] = np.array(self.manualExpCircSmooth)
            else:
                savedf['Circularity smooth'] = ''

            # savedf = self.expData[self.semPar] / (self.ymaxParManual - self.bParManual)
        savedf[''] = ''
        columns=[('Magnetic field','Gauss'),('Parallel component','arb. units'), ('Parallel SEM',''), ('Perpendicular component', 'arb. units'), ('Perpendicular component SEM',''), ('Difference', 'arb. units'),('Difference smooth', ''), ('Circularity', 'arb. units'),('Circularity smooth', ''), ('', '')]
        
        savedf.columns=pd.MultiIndex.from_tuples(columns)
        self.saveExpDataframe = savedf
        # self.saveExpDataframe.columns.set_levels('',level=1,inplace=True)
        print(savedf.head())


    def createSaveTheor(self):
        if len(self.theorData) <  len(self.theorData1):
            theorData = self.theorData1 
        else:
            theorData = self.theorData 
        savedf = pd.DataFrame(theorData, columns=['B', 'par_norm', 'per_norm', 'difference_norm', 'circularity'])
        self.saveTheorDataframe = savedf#.set_index('B')
        # print(savedf.head())


    def createSaveFile(self):
        self.saveExpDataframe = pd.DataFrame({('','') : pd.Series()})
        # self.saveExpDataframe=pd.MultiIndex.from_tuples([('','')])
        self.allTeor = pd.DataFrame({('','') : pd.Series()})
        # self.allTeor=pd.MultiIndex.from_tuples([('','')])
        if len(self.expData1)> 0:
            self.createSaveExp()
        if len(self.theoryDataList1) > 0:
            d = {}
            for i in range(len(self.theoryDataList)): 
                self.theorData1  = self.theoryDataList1[i]
                self.theorData  = self.theoryDataList[i]
                self.theoryFilenameLabel = self.theoryFilenamesLabels[i]
                self.createSaveTheor()
                d[self.theoryFilenameLabel] = self.saveTheorDataframe
            self.allTeor = pd.concat(d, axis=1)
            self.allTeor.columns = self.allTeor.columns.swaplevel(0, 1)
            for i in range(len(self.theoryDataList)):
                self.allTeor.insert(5*(i+1)+i, '', pd.Series(), allow_duplicates=True)

        if len(self.expData1) > 0:
            if len(self.theoryDataList1) > 0:
                self.saveAllDataframe = pd.concat([self.saveExpDataframe, self.allTeor], ignore_index=False, axis=1)
            else:
                self.saveAllDataframe = self.saveExpDataframe
        elif len(self.theoryDataList1) > 0:
            self.saveAllDataframe = self.allTeor
        else:
            pass

    def saveToCSV(self, savepath):
        savedir = os.path.dirname(savepath)
        # if not os.path.exists(savedir):
        #     os.makedirs(savedir)
        self.saveAllDataframe.to_csv(savepath, index=False)




if __name__ == '__main__':
    aoc = AOCcoefficients()
    # aoc.setRabi(np.arange(3,21))
    # aoc.setRabi(np.array([20,50,100]))#,1.0,2.0]))
    aoc.setRabi(np.array([1.0]))#,1.0,2.0]))
    pareja = [2, 3]
    jauda = 400
    aoc.setTheoryPath('/home/laima/Documents/Rb-2019/D1-merged_vienadasRabi/D1-Rb85-1/gamma=0.0180-DSteps=150-lWidth=2.0/{}-{}-pol=0-det=0/'.format(pareja[0], pareja[1]))

    exp_path = '/home/laima/Documents/Rb-2019/Rb-experiment/{}-{}/exp-lif-data/'.format(pareja[0], pareja[1])
    exp_name = 'Rb85_Fg{}_Fe{}_P={}uW_avg_smooth.csv'.format(pareja[0], pareja[1], jauda)
    experiment_filename = os.path.join(exp_path, exp_name)
    aoc.setExpFilename(experiment_filename)

    aoc.setTheoryFilenames(['/home/laima/Documents/Rb-2019/D1-merged_vienadasRabi/D1-Rb85-1/gamma=0.0180-DSteps=150-lWidth=2.0/{}-{}-pol=0-det=0/Rb85-D1-Rabi={:.2f}-shift=0.0-Ge=0'.format(pareja[0], pareja[1], Rabi)for Rabi in aoc.Rabi_list])

    aoc.setSavePath('/home/laima/Documents/Rb-2019/Rb-experiment/{}-{}/Best_fit_exp/'.format(pareja[0], pareja[1]))


    aoc.setMagneticCondition(50, 3000)

    # aoc.importExperimentalData()
    aoc.importAllTheoryDataList()

    # aoc.plotDiffExperiment()
    # aoc.plotCircExperiment()
    

    # aoc.createSaveFile()
    # cwd = os.getcwd()
    # aoc.saveToCSV(os.path.join(cwd+'/test', 'testTeor_exp_teor.csv'))
    # aoc.saveToCSV('/home/laima/Documents/AOC fitting parameters/AOC_fbs/src/main/python/test/testTeor_exp.csv')

    # from matplotlib.figure import Figure
    # from matplotlib.backends.backend_qt5agg import (
    # FigureCanvasQTAgg as FigureCanvas,
    # NavigationToolbar2QT as NavigationToolbar)

    fig = plt.figure()
    # canvas = FigureCanvas(fig)
    axPar = fig.add_subplot(221)
    axPer = fig.add_subplot(222)
    axDiff = fig.add_subplot(223)
    axCirc = fig.add_subplot(224)

    for kappa in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        aoc.kappa = kappa
        aoc.importAllTheoryDataList()
        print(aoc.theorData.head())
        
        aoc.theoryFilenameLabel = 'kappa = {}'.format(kappa)
        aoc.plotParallelTheory(axPar)
        aoc.plotPerpTheory(axPer)
        aoc.plotDiffTheory(axDiff)
        aoc.plotCircTheory(axCirc)




    # canvas.draw()
    axPar.legend()
    axPer.legend()
    axDiff.legend()
    axCirc.legend()
    plt.show()