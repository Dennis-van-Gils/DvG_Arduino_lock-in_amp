#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mishmash of PyQt5 stylesheets and custom controls that I personally use in
many of my projects.
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_PyQt_misc"
__date__        = "23-08-2018"
__version__     = "1.0.0"

from PyQt5 import QtWidgets as QtWid

COLOR_RED            = "rgb(255, 0, 0)"
COLOR_YELLOW         = "rgb(255, 255, 0)"
COLOR_INDIAN_RED     = "rgb(205, 92, 92)"
COLOR_SPRING_GREEN_2 = "rgb(0, 238, 118)"
COLOR_BISQUE_5       = "rgb(252, 208, 173)"
COLOR_READ_ONLY      = "rgb(250, 230, 210)"

# Retrieve the standard background color from
# QtGui.QApplication.palette().button().color().name()
# However, at init there is no QApplication instance yet and Python crashes
# hence hard-code here.
COLOR_BG = "rgb(240, 240, 240)"

# Style sheets

SS_TEXTBOX_READ_ONLY = (
        "QLineEdit {"
            "border: 1px solid black}"
        "QLineEdit::read-only {"
            "border: 1px solid gray;"
            "background: " + COLOR_READ_ONLY + "}"
        "QPlainTextEdit {"
            "border: 1px solid black}"
        "QPlainTextEdit[readOnly=\"true\"] {"
            "border: 1px solid gray;"
            "background-color: " + COLOR_READ_ONLY + "}")

SS_TEXTBOX_ERRORS = (
        "QLineEdit {"
            "border: 1px solid gray;"
            "background: " + COLOR_READ_ONLY + "}"
        "QLineEdit::read-only {"
            "border: 2px solid red;"
            "background: yellow;"
            "color: black}"
        "QPlainTextEdit {"
            "border: 1px solid gray;"
            "background-color: " + COLOR_READ_ONLY + "}"
        "QPlainTextEdit[readOnly=\"true\"] {"
            "border: 2px solid red;"
            "background-color: yellow;"
            "color: black}")

SS_GROUP = (
        "QGroupBox {"
            "background-color: " + COLOR_BISQUE_5 + ";"
            "border: 2px solid gray;"
            "border-radius: 5px;"
            "font: bold italic;"
            "padding: 8 0 0 0px;"
            "margin-top: 2ex}"
        "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "subcontrol-position: top center;"
            "padding: 0 3px}"
        "QGroupBox::flat {"
            "border: 0px;"
            "border-radius: 0 0px;"
            "padding: 0}")

SS_LED = (
        "QPushButton {"
            "background-color: " + COLOR_INDIAN_RED + ";"
            "border-style: inset;"
            "border-width: 1px;"
            "min-height: 30px;"
            "min-width: 30px;"
            "max-width: 30px}"
        "QPushButton::disabled {"
            "border-radius: 15px;"
            "color: black}"
        "QPushButton::checked {"
            "background-color: " + COLOR_SPRING_GREEN_2 + ";"
            "border-style: outset}")

SS_ERROR_LED = (
        "QPushButton {"
            "background-color: " + COLOR_SPRING_GREEN_2 + ";"
            "border: 1px solid gray;"
            "min-height: 30px;"
            "min-width: 30px}"
        "QPushButton::disabled {"
            "color: black}"
        "QPushButton::checked {"
            "font-weight: bold;"
            "background-color: red}")

SS_TINY_ERROR_LED = (
        "QPushButton {"
            "background-color: " + COLOR_BG + ";"
            "border-style: inset;"
            "border-width: 1px;"
            "max-height: 10px;"
            "max-width: 10px;"
            "height: 10px;"
            "width: 10px}"
        "QPushButton::disabled {"
            "border-radius: 5px;"
            "color: black}"
        "QPushButton::checked {"
            "background-color: red;"
            "border-style: outset}")

SS_TINY_LED = (
        "QPushButton {"
            "background-color: " + COLOR_BG + ";"
            "border-style: inset;"
            "border-width: 1px;"
            "max-height: 10px;"
            "max-width: 10px;"
            "height: 10px;"
            "width: 10px}"
        "QPushButton::disabled {"
            "border-radius: 5px;"
            "color: black}"
        "QPushButton::checked {"
            "background-color: " + COLOR_SPRING_GREEN_2 + ";"
            "border-style: outset}")

SS_TEXT_MSGS = (
        "QPlainTextEdit::disabled {"
            "color: black;"
            "background-color: white}")

SS_TITLE = (
        "QLabel {"
            "background-color: " + COLOR_BISQUE_5 + ";"
            "font: bold}")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def create_Toggle_button_style_sheet(bg_clr=COLOR_BG,
                                     color='black',
                                     checked_bg_clr=COLOR_SPRING_GREEN_2,
                                     checked_color='black',
                                     checked_font_weight='normal'):
    return ("QPushButton {"
                "background-color: " + bg_clr + ";" +
                "color: "            + color + ";" +
                "border-style: outset;"
                "border-width: 2px;"
                "border-radius: 4px;"
                "border-color: gray;"
                "text-align: center;"
                "padding: 1px 1px 1px 1px;}"
            "QPushButton:disabled {"
                "color: black;}"
            "QPushButton:checked {"
                "background-color: " + checked_bg_clr + ";"
                "color: "            + checked_color + ";" +
                "font-weight: "      + checked_font_weight + ";"
                "border-style: inset;}")

def create_LED_indicator():
    button = QtWid.QPushButton("0", checkable=True, enabled=False)
    button.setStyleSheet(SS_LED)
    return button

def create_Relay_button():
    button = QtWid.QPushButton("0", checkable=True)
    button.setStyleSheet(SS_LED)
    return button

def create_Toggle_button(text='', minimumHeight=40):
    SS = ("QPushButton {"
              "background-color: " + COLOR_BG + ";" +
              "color: black;" +
              "border-style: outset;"
              "border-width: 2px;"
              "border-radius: 4px;"
              "border-color: gray;"
              "text-align: center;"
              "padding: 1px 1px 1px 1px;}"
          "QPushButton:disabled {"
              "color: grey;}"
          "QPushButton:checked {"
              "background-color: " + COLOR_SPRING_GREEN_2 + ";"
              "color: black;" +
              "font-weight: normal;"
              "border-style: inset;}")
    button = QtWid.QPushButton(text, checkable=True)
    button.setStyleSheet(SS)

    if minimumHeight is not None:
        button.setMinimumHeight(minimumHeight)

    return button

def create_Toggle_button_2(text='', minimumHeight=40):
    SS = ("QPushButton {"
              "background-color: " + COLOR_BG + ";" +
              "color: black;" +
              "border-style: outset;"
              "border-width: 2px;"
              "border-radius: 4px;"
              "border-color: gray;"
              "text-align: center;"
              "padding: 1px 1px 1px 1px;}"
          "QPushButton:disabled {"
              "color: grey;}"
          "QPushButton:checked {"
              "background-color: " + COLOR_YELLOW + ";"
              "color: black;" +
              "font-weight: bold;"
              "border-color: red;"
              "border-style: inset;}")
    button = QtWid.QPushButton(text, checkable=True)
    button.setStyleSheet(SS)

    if minimumHeight is not None:
        button.setMinimumHeight(minimumHeight)

    return button

def create_Toggle_button_3(text='', minimumHeight=40):
    SS = ("QPushButton {"
              "background-color: " + COLOR_YELLOW + ";" +
              "color: black;" +
              "border-style: outset;"
              "border-width: 2px;"
              "border-radius: 4px;"
              "border-color: red;"
              "text-align: center;"
              "font-weight: bold;"
              "padding: 1px 1px 1px 1px;}"
          "QPushButton:disabled {"
              "color: grey;}"
          "QPushButton:checked {"
              "background-color: " + COLOR_SPRING_GREEN_2 + ";"
              "color: black;" +
              "border-color: gray;"
              "border-style: inset;"
              "font-weight: normal;}")
    button = QtWid.QPushButton(text, checkable=True)
    button.setStyleSheet(SS)

    if minimumHeight is not None:
        button.setMinimumHeight(minimumHeight)

    return button

def create_tiny_error_LED(text=''):
    button = QtWid.QPushButton(text='', checkable=True, enabled=False)
    button.setStyleSheet(SS_TINY_ERROR_LED)
    return button

def create_tiny_LED(text=''):
    button = QtWid.QPushButton(text='', checkable=True, enabled=False)
    button.setStyleSheet(SS_TINY_LED)
    return button

def create_error_LED(text=''):
    button = QtWid.QPushButton(text='', checkable=True, enabled=False)
    button.setStyleSheet(SS_ERROR_LED)
    return button
