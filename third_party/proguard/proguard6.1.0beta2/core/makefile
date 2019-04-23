# GNU/Linux makefile for ProGuard.

include build.properties

GSON_VERSION = 2.8.5
GSON_URL     = https://jcenter.bintray.com/com/google/code/gson/gson/${GSON_VERSION}/gson-${GSON_VERSION}.jar
GSON_JAR     = $(LIB)/gson-${GSON_VERSION}.jar

MAIN_CLASS       = proguard/ProGuard
CLASSPATH        = $(GSON_JAR)
TARGET           = proguard
INCLUDE_MANIFEST = true

include ../buildscripts/functions.mk

$(GSON_JAR):
	$(DOWNLOAD) $(GSON_JAR) $(GSON_URL)
