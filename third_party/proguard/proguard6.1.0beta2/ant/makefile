# GNU/Linux makefile for the ProGuard Ant task.

ifndef ANT_HOME
ANT_HOME = /usr/local/java/ant
endif

MAIN_CLASS = proguard/ant/ProGuardTask
ANT_JAR    = $(ANT_HOME)/lib/ant.jar
CLASSPATH  = ../core/$(OUT):$(ANT_JAR)
TARGET     = proguard
UPDATE_JAR = true

include ../buildscripts/functions.mk

$(ANT_JAR): ; $(error Please make sure ANT_HOME is set correctly)
