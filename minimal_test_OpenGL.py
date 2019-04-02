# -*- coding: utf-8 -*-
"""
Minimal test case for OpenGL hardware support in pyqtgraph.

The official PyOpenGL library contains type errors on Win10 platforms. It will
try to use type 'float128' which should be 'longdouble'.

You must use the unofficial OpenGL and OpenGL-accelerate modules by
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl

PyOpenGL-3.1.3b2-cp37-cp37m-win_amd64.whl
PyOpenGL_accelerate-3.1.3b2-cp37-cp37m-win_amd64.whl

Dennis van Gils
02-04-2019
"""
import os
import sys

from PyQt5 import QtCore
from PyQt5.QtCore import QSize
from PyQt5 import QtWidgets as QtWid

import OpenGL.GL as gl
import pyqtgraph as pg
import numpy as np

USE_OPENGL = True
if USE_OPENGL:
    pg.setConfigOptions(useOpenGL=True)
    pg.setConfigOptions(enableExperimental=True)
    pg.setConfigOptions(antialias=False)
    
    # Monkey patch error in pyqtgraph
    import DvG_monkeypatch_pyqtgraph as pgmp
    pg.PlotCurveItem.paintGL = pgmp.PlotCurveItem_paintGL
else:
    pg.setConfigOptions(useOpenGL=False)
    pg.setConfigOptions(enableExperimental=True)
    pg.setConfigOptions(antialias=False)

@QtCore.pyqtSlot()
def getOpenglInfo():
    """Seems only possible to be called after window.show(). I guess OpenGL has
    not initialized yet properly before window.show()
    """
    info = ("  Vendor  : %s\n"
            "  Renderer: %s\n"
            "  OpenGL  : %s\n"
            "  Shader  : %s\n" % (
                    gl.glGetString(gl.GL_VENDOR).decode(),
                    gl.glGetString(gl.GL_RENDERER).decode(),
                    gl.glGetString(gl.GL_VERSION).decode(),
                    gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode()))
    return info

class Window(QtWid.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        #self.setGeometry(800, 400, 800, 600)
        self.setWindowTitle("PyOpenGL demo in pyqtgraph")
        
        #self.glWidget = GLWidget()
        
        num_points = int(5e3)
        x = np.array(np.arange(num_points), dtype=float)
        y = np.random.rand(num_points)
        
        self.gw = pg.GraphicsWindow()
        self.pi = self.gw.addPlot()
        self.pi.plot(x=x, y=y, pen=pg.mkPen(color=[255, 30 , 180], width=3))
        self.pi.setAutoVisible(x=True, y=True)
        self.pi.enableAutoRange('x', False)
        self.pi.enableAutoRange('y', False)
        self.pi.setClipToView(True)
        self.pi.setTitle('PyOpenGL support = %s' % USE_OPENGL)
        
        self.qpbt_info = QtWid.QPushButton("OpenGL info")
        self.qpbt_info.clicked.connect(lambda: print(getOpenglInfo()))
        
        main_layout = QtWid.QHBoxLayout()
        #main_layout.addWidget(self.glWidget)
        main_layout.addWidget(self.gw)
        main_layout.addWidget(self.qpbt_info)
        self.setLayout(main_layout)

class GLWidget(QtWid.QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
    
    def initializeGL(self):
        print(getOpenglInfo())
        gl.glEnable(gl.GL_STENCIL_TEST)
        
    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

if __name__ == '__main__':
    print("PID: %s\n" % os.getpid())
    
    app = 0
    app = QtWid.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())