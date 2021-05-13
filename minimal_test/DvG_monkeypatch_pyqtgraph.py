#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monkey patches for pyqtgraph.
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = ""
__date__        = "02-04-2019"
__version__     = "1.0.0"

from PyQt5 import QtGui, QtCore
import numpy as np
import pyqtgraph as fn

def PlotCurveItem_paintGL(self, p, opt, widget):
    """
    Edited:
        pyqtgraph/graphicsItems/PlotCurveItem.py
        v0.10
    
    Line 566 in 2e69b9c
    Reads:
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, pos.size / pos.shape[-1])
    Fixed TypeError:
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, int(pos.size / pos.shape[-1]))
    
    Added missing statement between line 561 and 562 in 2e69b9c:
        gl.glLineWidth(width)
    """
    p.beginNativePainting()
    import OpenGL.GL as gl
    
    ## set clipping viewport
    view = self.getViewBox()
    if view is not None:
        rect = view.mapRectToItem(self, view.boundingRect())
        #gl.glViewport(int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        
        #gl.glTranslate(-rect.x(), -rect.y(), 0)
        
        gl.glEnable(gl.GL_STENCIL_TEST)
        gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE) # disable drawing to frame buffer
        gl.glDepthMask(gl.GL_FALSE)  # disable drawing to depth buffer
        gl.glStencilFunc(gl.GL_NEVER, 1, 0xFF)  
        gl.glStencilOp(gl.GL_REPLACE, gl.GL_KEEP, gl.GL_KEEP)  
        
        ## draw stencil pattern
        gl.glStencilMask(0xFF)
        gl.glClear(gl.GL_STENCIL_BUFFER_BIT)
        gl.glBegin(gl.GL_TRIANGLES)
        gl.glVertex2f(rect.x(), rect.y())
        gl.glVertex2f(rect.x()+rect.width(), rect.y())
        gl.glVertex2f(rect.x(), rect.y()+rect.height())
        gl.glVertex2f(rect.x()+rect.width(), rect.y()+rect.height())
        gl.glVertex2f(rect.x()+rect.width(), rect.y())
        gl.glVertex2f(rect.x(), rect.y()+rect.height())
        gl.glEnd()
                   
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glStencilMask(0x00)
        gl.glStencilFunc(gl.GL_EQUAL, 1, 0xFF)
        
    try:
        x, y = self.getData()
        pos = np.empty((len(x), 2))
        pos[:,0] = x
        pos[:,1] = y
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        try:
            gl.glVertexPointerf(pos)
            pen = fn.mkPen(self.opts['pen'])
            color = pen.color()
            gl.glColor4f(color.red()/255., color.green()/255., color.blue()/255., color.alpha()/255.)
            width = pen.width()
            if pen.isCosmetic() and width < 1:
                width = 1
            gl.glPointSize(width)
            gl.glLineWidth(width)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gl.glDrawArrays(gl.GL_LINE_STRIP, 0, int(pos.size / pos.shape[-1]))
        finally:
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
    finally:
        p.endNativePainting()

def ImageExporter_export(self, fileName=None, toBytes=False, copy=False):
    """
    Edited:
        pyqtgraph/exporters/ImageExporter.py
        v0.10
    
    Lines 67 & 70 in f4e0f09
    Read:
        w, h = self.params['width'], self.params['height']
        bg = np.empty((self.params['width'], self.params['height'], 4), dtype=np.ubyte)
    Fixed TypeError:
        w, h = int(self.params['width']), int(self.params['height'])
        bg = np.empty((w, h, 4), dtype=np.ubyte)
    """

    if fileName is None and not toBytes and not copy:
        #if USE_PYSIDE:
        #    filter = ["*."+str(f) for f in QtGui.QImageWriter.supportedImageFormats()]
        #else:
        filter = ["*."+bytes(f).decode('utf-8') for f in QtGui.QImageWriter.supportedImageFormats()]
        preferred = ['*.png', '*.tif', '*.jpg']
        for p in preferred[::-1]:
            if p in filter:
                filter.remove(p)
                filter.insert(0, p)
        self.fileSaveDialog(filter=filter)
        return
        
    targetRect = QtCore.QRect(0, 0, self.params['width'], self.params['height'])
    sourceRect = self.getSourceRect()
    
    
    #self.png = QtGui.QImage(targetRect.size(), QtGui.QImage.Format_ARGB32)
    #self.png.fill(pyqtgraph.mkColor(self.params['background']))
    w, h = int(self.params['width']), int(self.params['height'])
    if w == 0 or h == 0:
        raise Exception("Cannot export image with size=0 (requested export size is %dx%d)" % (w,h))
    bg = np.empty((w, h, 4), dtype=np.ubyte)
    color = self.params['background']
    bg[:,:,0] = color.blue()
    bg[:,:,1] = color.green()
    bg[:,:,2] = color.red()
    bg[:,:,3] = color.alpha()
    self.png = fn.makeQImage(bg, alpha=True)
    
    ## set resolution of image:
    origTargetRect = self.getTargetRect()
    resolutionScale = targetRect.width() / origTargetRect.width()
    #self.png.setDotsPerMeterX(self.png.dotsPerMeterX() * resolutionScale)
    #self.png.setDotsPerMeterY(self.png.dotsPerMeterY() * resolutionScale)
    
    painter = QtGui.QPainter(self.png)
    #dtr = painter.deviceTransform()
    try:
        self.setExportMode(True, {'antialias': self.params['antialias'], 'background': self.params['background'], 'painter': painter, 'resolutionScale': resolutionScale})
        painter.setRenderHint(QtGui.QPainter.Antialiasing, self.params['antialias'])
        self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))
    finally:
        self.setExportMode(False)
    painter.end()
    
    if copy:
        QtGui.QApplication.clipboard().setImage(self.png)
    elif toBytes:
        return self.png
    else:
        self.png.save(fileName)