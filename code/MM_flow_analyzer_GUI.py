from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QColor
import sys, os
from GUI.CellMate_main import Ui_UiMain as Ui_Main
import MM_flow_analyzer



class EmittingStream(QObject):
    # Captures stdout for the GUI-log
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class ErrorEmittingStream(QObject):
    # Captures stderr for the GUI-log
    textWritten = pyqtSignal(str)

    def write(self, errortext):
        self.textWritten.emit(str(errortext))


class AnalysisThread(QThread):
    # Runs the analysis in a separate thread in order not to freeze the GUI

    def __init__(self, fnamelist, outdir, flowkwargs, scalebarFlag, scalebarLength, channel, unit):
        QThread.__init__(self)

        self.fnamelist = fnamelist
        self.flowkwargs = flowkwargs
        self.outdir = outdir
        self.scalebarFlag = scalebarFlag
        self.scalebarLength = scalebarLength
        self.channel = channel - 1
        self.unit = unit

    def __del__(self):
        self.quit()

    def run(self):

        MM_flow_analyzer.analyzeFiles(self.fnamelist, self.outdir, self.flowkwargs,
                                      self.scalebarFlag, self.scalebarLength,
                                      self.channel, self.unit)


class AppWindow(QDialog, Ui_Main):
    def __init__(self):
        super(AppWindow, self).__init__()

        self.setupUi(self)

        self.listWidget_filesToAnalyze.addItem("No files selected")

        #Make buttons work
        self.btn_selectFilesToAnalyze.clicked.connect(self.in_folder_select)
        self.btn_selectOutDir.clicked.connect(self.out_folder_select)
        self.btn_run.clicked.connect(self._analyzeFiles)
        #self.btn_cancel.clicked().connect(self.close)

        # Installs custom output streams
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = ErrorEmittingStream(textWritten=self.errorOutputWritten)

        self.show()

    def __del__(self):
        # Restore sys.stdout and sys.stderr
        #pass
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def in_folder_select(self):

        directory = QFileDialog.getExistingDirectory(self, "Pick a folder")
        self.infiles = []
        if directory:
            self.listWidget_filesToAnalyze.clear()
            for fname in os.listdir(directory):
                if fname.endswith(".tif"):
                    self.infiles.append(os.path.join(directory, fname))
                    self.listWidget_filesToAnalyze.addItem(fname)

    def out_folder_select(self):
        directory = QFileDialog.getExistingDirectory(self, "Pick a folder")

        if directory:
            self.listWidget_outDir.clear()
            self.listWidget_outDir.addItem(directory)
            self.outdir = directory

    def _analyzeFiles(self):

        self.separateThread = AnalysisThread(self.infiles, self.outdir,
                                             self._getFlowkwargs(), self._getScaleBarFlag(), self._getScaleBarLength(),
                                             self._getChannel(), self._getUnit())

        self.separateThread.start()

    def _getChannel(self):

        return (int(self.channel_selectBox.value()))

    def _getFlowkwargs(self):

        flowkwargs = {}

        flowkwargs["step"] = self.distance.value()
        flowkwargs["scale"] = self.vector_scaler.value()
        flowkwargs["line_thicknes"] = self.lineThickens.value()

        return flowkwargs

    def _getScaleBarFlag(self):

        return self.scalebar_ceckBox.isChecked()

    def _getScaleBarLength(self):

        return self.scalebarLength_box.value()

    def _getUnit(self):

        return str(self.output_unitBox.currentText())

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""

        self.logOutput.insertPlainText(text)
        sb = self.logOutput.verticalScrollBar()
        sb.setValue(sb.maximum())


    def errorOutputWritten(self, errortext):
        """Append red error text to the QTextEdit."""

        # sets fontcolor to red for warnings
        color = QColor(255, 0, 0)
        self.logOutput.setTextColor(color)

        # Write ouput to log
        self.logOutput.insertPlainText(errortext)

        # Set fontcolor back to black
        color = QColor(0, 0, 0)
        self.logOutput.setTextColor(color)

        # Autoscroll the text
        sb = self.logOutput.verticalScrollBar()
        sb.setValue(sb.maximum())


def main():
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':  # if we're running file directly and not importing it
    main()  # run the main function
