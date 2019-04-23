# GNU/Linux makefile for the ProGuard Wireless Toolkit plugin.

ifndef WTK_HOME
WTK_HOME = /usr/local/java/wtk
endif

MAIN_CLASS = proguard/wtk/ProGuardObfuscator
WTK_JAR    = $(WTK_HOME)/wtklib/kenv.zip
CLASSPATH  = ../core/$(OUT):$(WTK_JAR)
TARGET     = proguard
UPDATE_JAR = true

include ../buildscripts/functions.mk

$(WTK_JAR): ; $(error Please make sure WTK_HOME is set correctly)
