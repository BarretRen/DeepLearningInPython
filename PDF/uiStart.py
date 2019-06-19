import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot
from Ui_bookmark import Ui_MainWindow  # 导入创建的GUI类
from addBookmarkToPDF import MyPDFHandler, PDFHandleMode


# 自己建一个mywindows类，mywindow是自己的类名。QtWidgets.QMainWindow：继承该类方法
class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    pdfpath = ''
    txtpath = ''
    outpath = ''

    # __init__:析构函数，也就是类被创建后就会预先加载的项目。
    #  马上运行，这个方法可以用来对你的对象做一些你希望的初始化。
    def __init__(self):
        # 这里需要重载一下mywindow，同时也包含了QtWidgets.QMainWindow的预加载项。
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.toolButton.clicked.connect(self.pdfclick)
        self.toolButton_2.clicked.connect(self.txtclick)
        self.toolButton_3.clicked.connect(self.outputclick)
        self.pushButton.clicked.connect(self.pushButtonclick)

    @pyqtSlot()
    def pdfclick(self):
        self.pdfpath = ''

    @pyqtSlot()
    def txtclick(self):
        self.txtpath = ''

    @pyqtSlot()
    def outputclick(self):
        self.outpath = ''

    @pyqtSlot()
    def pushButtonclick(self):
        pdf_handler = MyPDFHandler(self.pdfpath, PDFHandleMode.NEWLY)
        pdf_handler.add_bookmarks_by_read_txt(self.txtpath)
        pdf_handler.save2file(self.outpath)


if __name__ == '__main__':  # 如果整个程序是主程序
    # QApplication相当于main函数，也就是整个程序（很多文件）的主入口函数。
    # 对于GUI程序必须至少有一个这样的实例来让程序运行。
    app = QtWidgets.QApplication(sys.argv)
    # 生成 mywindow 类的实例。
    window = mywindow()
    # 有了实例，就得让它显示，show()是QWidget的方法，用于显示窗口。
    window.show()
    # 调用sys库的exit退出方法，条件是app.exec_()，也就是整个窗口关闭。
    # 有时候退出程序后，sys.exit(app.exec_())会报错，改用app.exec_()就没事
    # https://stackoverflow.com/questions/25719524/difference-between-sys-exitapp-exec-and-app-exec
    sys.exit(app.exec_())
