# GNU/Linux makefile for ProGuard.

ANT_HOME    = /usr/local/java/ant
GRADLE_HOME = /usr/local/java/gradle
WTK_HOME    = /usr/local/java/wtk

PROGUARD_HOME := $(subst ./..,..,$(subst /buildscripts/..,/,$(dir $(MAKEFILE_LIST))..))
SRC     = $(PROGUARD_HOME)/src
CLASSES = $(PROGUARD_HOME)/classes
LIB     = $(PROGUARD_HOME)/lib

ANT_JAR     = $(ANT_HOME)/lib/ant.jar
GRADLE_JARS = $(wildcard \
              $(GRADLE_HOME)/lib/plugins/gradle-plugins-*.jar \
              $(GRADLE_HOME)/lib/gradle-logging-*.jar \
              $(GRADLE_HOME)/lib/gradle-base-services-*.jar \
              $(GRADLE_HOME)/lib/gradle-base-services-groovy-*.jar \
              $(GRADLE_HOME)/lib/gradle-core-*.jar \
              $(GRADLE_HOME)/lib/groovy-all-*.jar)
WTK_JAR     = $(WTK_HOME)/wtklib/kenv.zip

NOTHING:=
SPACE:=$(NOTHING) $(NOTHING)
CLASSPATH = $(ANT_JAR):$(subst $(SPACE),:,$(GRADLE_JARS)):$(WTK_JAR)

PROGUARD     = proguard/ProGuard
PROGUARD_GUI = proguard/gui/ProGuardGUI
RETRACE      = proguard/retrace/ReTrace
ANT_TASK     = proguard/ant/ProGuardTask
GRADLE_TASK  = proguard/gradle/ProGuardTask
WTK_PLUGIN   = proguard/wtk/ProGuardObfuscator

TARGETS = $(PROGUARD) $(PROGUARD_GUI) $(RETRACE) $(ANT_TASK) $(GRADLE_TASK) $(WTK_PLUGIN)

JAVAC_OPTIONS = -nowarn -Xlint:none -classpath $(CLASSPATH) -sourcepath $(SRC) -d $(CLASSES)

# Command sequence definitions for creating jars.

define createjar
  jar -cfm $(LIB)/$@.jar $(SRC)/$(dir $<)MANIFEST.MF \
    -C $(CLASSES) $(dir $<)
endef

define updatejar
  jar -uf $(LIB)/proguard.jar \
    -C $(CLASSES) $(dir $<)
endef

# The various targets.

all:     basic options
basic:   proguard proguardgui retrace
options: anttask gradletask wtkplugin

proguard: $(PROGUARD)
	$(createjar)

proguardgui: proguard
proguardgui: $(PROGUARD_GUI)
	$(createjar)

retrace: $(RETRACE)
	$(createjar)

anttask: $(ANT_JAR)
anttask: $(PROGUARD)
anttask: $(ANT_TASK)
	$(updatejar)

gradletask: $(GRADLE_JARS)
gradletask: $(PROGUARD)
gradletask: $(GRADLE_TASK)
	$(updatejar)

wtkplugin: $(WTK_JAR)
wtkplugin: $(PROGUARD)
wtkplugin: $(WTK_PLUGIN)
	$(updatejar)

clean:
	-rm -fr $(CLASSES) $(LIB)


define RESOURCES
  $(shell find $(SRC)/$(dir $(1)) -maxdepth 1 \( -name \*.properties -o -name \*.png -o -name \*.gif -o -name \*.pro \) -printf $(CLASSES)/$(dir $(1))%P\\n)
endef

define TARGETRULE
  $(1): $(CLASSES) $(CLASSES)/$(1).class $(call RESOURCES,$(1)) $(LIB)
endef

$(foreach TARGET,$(TARGETS),$(eval $(call TARGETRULE,$(TARGET))))

$(CLASSES) $(LIB):
	-mkdir -p $@

$(CLASSES)/%.class: $(SRC)/%.java
	javac $(JAVAC_OPTIONS) $^

$(CLASSES)/%.properties $(CLASSES)/%.png $(CLASSES)/%.gif $(CLASSES)/%.pro:
	cp $(subst $(CLASSES),$(SRC),$@) $@

%.jar %.zip:
	@echo "Please make sure the path to $@ is set"
	@echo "correctly in this $(strip $(MAKEFILE_LIST))."
	@echo "Alternatively, if you don't need the corresponding option,"
	@echo "you can run `make' with the option -k."

.PHONY: all basic options proguard proguardgui retrace anttask wtkplugin clean $(TARGETS) $(OPTIONAL_TARGETS)
