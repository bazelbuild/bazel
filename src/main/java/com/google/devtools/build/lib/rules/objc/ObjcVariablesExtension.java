// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.errorprone.annotations.CanIgnoreReturnValue;

/** Build variable extensions for templating a toolchain for objc builds. */
class ObjcVariablesExtension implements VariablesExtension {

  static final String PCH_FILE_VARIABLE_NAME = "pch_file";
  static final String FRAMEWORKS_PATH_NAME = "framework_paths";
  static final String OBJC_MODULE_CACHE_DIR_NAME = "_objc_module_cache";
  static final String OBJC_MODULE_CACHE_KEY = "modules_cache_path";
  static final String ARCHIVE_PATH_VARIABLE_NAME = "archive_path";
  static final String LINKMAP_EXEC_PATH = "linkmap_exec_path";

  // executable linking variables
  static final String FRAMEWORK_NAMES_VARIABLE_NAME = "framework_names";
  static final String WEAK_FRAMEWORK_NAMES_VARIABLE_NAME = "weak_framework_names";
  static final String LIBRARY_NAMES_VARIABLE_NAME = "library_names";
  static final String FILELIST_VARIABLE_NAME = "filelist";
  static final String LINKED_BINARY_VARIABLE_NAME = "linked_binary";
  static final String FORCE_LOAD_EXEC_PATHS_VARIABLE_NAME = "force_load_exec_paths";
  static final String DEP_LINKOPTS_VARIABLE_NAME = "dep_linkopts";
  static final String ATTR_LINKOPTS_VARIABLE_NAME = "attr_linkopts";

  // dsym variables
  static final String DSYM_PATH_VARIABLE_NAME = "dsym_path";

  // ARC variables. Mutually exclusive.
  static final String OBJC_ARC_VARIABLE_NAME = "objc_arc";
  static final String NO_OBJC_ARC_VARIABLE_NAME = "no_objc_arc";

  private final RuleContext ruleContext;
  private final IntermediateArtifacts intermediateArtifacts;
  private final BuildConfigurationValue buildConfiguration;
  private final ImmutableList<String> frameworkSearchPaths;
  private final ImmutableList<String> frameworkNames;
  private final ImmutableList<String> weakFrameworkNames;
  private final ImmutableList<String> libraryNames;
  private final ImmutableSet<Artifact> forceLoadArtifacts;
  private final ImmutableList<String> depLinkopts;
  private final ImmutableList<String> attributeLinkopts;
  private final ImmutableSet<VariableCategory> activeVariableCategories;
  private final Artifact dsymSymbol;
  private final Artifact linkmap;
  private boolean arcEnabled = true;

  private ObjcVariablesExtension(
      RuleContext ruleContext,
      IntermediateArtifacts intermediateArtifacts,
      BuildConfigurationValue buildConfiguration,
      ImmutableList<String> frameworkSearchPaths,
      ImmutableList<String> frameworkNames,
      ImmutableList<String> weakFrameworkNames,
      ImmutableList<String> libraryNames,
      ImmutableSet<Artifact> forceLoadArtifacts,
      ImmutableList<String> depLinkopts,
      ImmutableList<String> attributeLinkopts,
      ImmutableSet<VariableCategory> activeVariableCategories,
      Artifact dsymSymbol,
      Artifact linkmap,
      boolean arcEnabled) {
    this.ruleContext = ruleContext;
    this.intermediateArtifacts = intermediateArtifacts;
    this.buildConfiguration = buildConfiguration;
    this.frameworkSearchPaths = frameworkSearchPaths;
    this.frameworkNames = frameworkNames;
    this.weakFrameworkNames = weakFrameworkNames;
    this.libraryNames = libraryNames;
    this.forceLoadArtifacts = forceLoadArtifacts;
    this.depLinkopts = depLinkopts;
    this.attributeLinkopts = attributeLinkopts;
    this.activeVariableCategories = activeVariableCategories;
    this.dsymSymbol = dsymSymbol;
    this.linkmap = linkmap;
    this.arcEnabled = arcEnabled;
  }

  /** Type of build variable that can optionally exported by this extension. */
  public enum VariableCategory {
    EXECUTABLE_LINKING_VARIABLES,
    DSYM_VARIABLES,
    LINKMAP_VARIABLES,
    MODULE_MAP_VARIABLES
  }

  @Override
  public void addVariables(CcToolchainVariables.Builder builder) {
    addPchVariables(builder);
    if (activeVariableCategories.contains(VariableCategory.MODULE_MAP_VARIABLES)) {
      addModuleMapVariables(builder);
    }
    if (activeVariableCategories.contains(VariableCategory.EXECUTABLE_LINKING_VARIABLES)) {
      addExecutableLinkVariables(builder);
    }
    if (activeVariableCategories.contains(VariableCategory.DSYM_VARIABLES)) {
      addDsymVariables(builder);
    }
    if (activeVariableCategories.contains(VariableCategory.LINKMAP_VARIABLES)) {
      addLinkmapVariables(builder);
    }
    if (arcEnabled) {
      builder.addStringVariable(OBJC_ARC_VARIABLE_NAME, "");
    } else {
      builder.addStringVariable(NO_OBJC_ARC_VARIABLE_NAME, "");
    }
  }

  private void addPchVariables(CcToolchainVariables.Builder builder) {
    if (ruleContext.attributes().has("pch", BuildType.LABEL)
        && ruleContext.getPrerequisiteArtifact("pch") != null) {
      builder.addStringVariable(
          PCH_FILE_VARIABLE_NAME, ruleContext.getPrerequisiteArtifact("pch").getExecPathString());
    }
  }

  private void addModuleMapVariables(CcToolchainVariables.Builder builder) {
    builder.addStringVariable(
        OBJC_MODULE_CACHE_KEY,
        buildConfiguration.getGenfilesFragment(ruleContext.getRepository())
            + "/"
            + OBJC_MODULE_CACHE_DIR_NAME);
  }

  private void addExecutableLinkVariables(CcToolchainVariables.Builder builder) {
    builder.addStringSequenceVariable(FRAMEWORKS_PATH_NAME, frameworkSearchPaths);
    builder.addStringSequenceVariable(
        FRAMEWORK_NAMES_VARIABLE_NAME, frameworkNames);
    builder.addStringSequenceVariable(WEAK_FRAMEWORK_NAMES_VARIABLE_NAME, weakFrameworkNames);
    builder.addStringSequenceVariable(LIBRARY_NAMES_VARIABLE_NAME, libraryNames);
    builder.addStringVariable(
        FILELIST_VARIABLE_NAME, intermediateArtifacts.linkerObjList().getExecPathString());
    builder.addStringVariable(
        LINKED_BINARY_VARIABLE_NAME,
        ruleContext.getFragment(CppConfiguration.class).objcShouldStripBinary()
            ? intermediateArtifacts.unstrippedSingleArchitectureBinary().getExecPathString()
            : intermediateArtifacts.strippedSingleArchitectureBinary().getExecPathString());

    builder.addStringSequenceVariable(
        FORCE_LOAD_EXEC_PATHS_VARIABLE_NAME,
        Artifact.toExecPaths(forceLoadArtifacts));
    builder.addStringSequenceVariable(DEP_LINKOPTS_VARIABLE_NAME, depLinkopts);
    builder.addStringSequenceVariable(ATTR_LINKOPTS_VARIABLE_NAME, attributeLinkopts);
  }

  private static String getShellEscapedExecPathString(Artifact artifact) {
    return ShellUtils.shellEscape(artifact.getExecPathString());
  }

  private void addDsymVariables(CcToolchainVariables.Builder builder) {
    builder.addStringVariable(DSYM_PATH_VARIABLE_NAME, getShellEscapedExecPathString(dsymSymbol));
  }

  private void addLinkmapVariables(CcToolchainVariables.Builder builder) {
    builder.addStringVariable(LINKMAP_EXEC_PATH, linkmap.getExecPathString());
  }

  /** A Builder for {@link ObjcVariablesExtension}. */
  static class Builder {
    private RuleContext ruleContext;
    private IntermediateArtifacts intermediateArtifacts;
    private BuildConfigurationValue buildConfiguration;
    private ImmutableList<String> frameworkSearchPaths;
    private ImmutableList<String> frameworkNames;
    private ImmutableList<String> weakFrameworkNames;
    private ImmutableSet<Artifact> forceLoadArtifacts;
    private ImmutableList<String> libraryNames;
    private ImmutableList<String> depLinkopts;
    private ImmutableList<String> attributeLinkopts;
    private Artifact dsymSymbol;
    private Artifact linkmap;
    private boolean arcEnabled = true;

    private final ImmutableSet.Builder<VariableCategory> activeVariableCategoriesBuilder =
        ImmutableSet.builder();

    /** Sets the {@link RuleContext} for this extension. */
    @CanIgnoreReturnValue
    public Builder setRuleContext(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
      return this;
    }

    /** Sets the {@link IntermediateArtifacts} for this extension. */
    @CanIgnoreReturnValue
    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = Preconditions.checkNotNull(intermediateArtifacts);
      return this;
    }

    /** Sets the configuration for this extension. */
    @CanIgnoreReturnValue
    public Builder setConfiguration(BuildConfigurationValue buildConfiguration) {
      this.buildConfiguration = Preconditions.checkNotNull(buildConfiguration);
      return this;
    }

    /** Sets the framework search paths to be passed to the compiler/linker using {@code -F}. */
    @CanIgnoreReturnValue
    public Builder setFrameworkSearchPath(ImmutableList<String> frameworkSearchPaths) {
      this.frameworkSearchPaths = Preconditions.checkNotNull(frameworkSearchPaths);
      return this;
    }

    /** Sets the framework names to be passed to the linker using {@code -framework}. */
    @CanIgnoreReturnValue
    public Builder setFrameworkNames(ImmutableList<String> frameworkNames) {
      this.frameworkNames = Preconditions.checkNotNull(frameworkNames);
      return this;
    }

    /** Sets the weak framework names to be passed to the linker using {@code -weak_framework}. */
    @CanIgnoreReturnValue
    public Builder setWeakFrameworkNames(ImmutableList<String> weakFrameworkNames) {
      this.weakFrameworkNames = Preconditions.checkNotNull(weakFrameworkNames);
      return this;
    }

    /** Sets binary input files to be passed to the linker with "-l" flags. */
    @CanIgnoreReturnValue
    public Builder setLibraryNames(ImmutableList<String> libraryNames) {
      this.libraryNames = Preconditions.checkNotNull(libraryNames);
      return this;
    }

    /** Sets artifacts to be passed to the linker with {@code -force_load}. */
    @CanIgnoreReturnValue
    public Builder setForceLoadArtifacts(ImmutableSet<Artifact> forceLoadArtifacts) {
      this.forceLoadArtifacts = Preconditions.checkNotNull(forceLoadArtifacts);
      return this;
    }

    /** Sets linkopts from dependency. */
    @CanIgnoreReturnValue
    public Builder setDepLinkopts(ImmutableList<String> depLinkopts) {
      this.depLinkopts = Preconditions.checkNotNull(depLinkopts);
      return this;
    }

    /** Sets linkopts arising from rule attributes. */
    @CanIgnoreReturnValue
    public Builder setAttributeLinkopts(ImmutableList<String> attributeLinkopts) {
      this.attributeLinkopts = Preconditions.checkNotNull(attributeLinkopts);
      return this;
    }

    /** Sets the given {@link VariableCategory} as active for this extension. */
    @CanIgnoreReturnValue
    public Builder addVariableCategory(VariableCategory variableCategory) {
      this.activeVariableCategoriesBuilder.add(Preconditions.checkNotNull(variableCategory));
      return this;
    }

    /** Sets the Artifact for the dsym symbol file. */
    @CanIgnoreReturnValue
    public Builder setDsymSymbol(Artifact dsymSymbol) {
      this.dsymSymbol = dsymSymbol;
      return this;
    }

    /** Sets the Artifact for the linkmap. */
    @CanIgnoreReturnValue
    public Builder setLinkmap(Artifact linkmap) {
      this.linkmap = linkmap;
      return this;
    }

    /** Sets whether ARC is enabled. */
    @CanIgnoreReturnValue
    public Builder setArcEnabled(boolean enabled) {
      this.arcEnabled = enabled;
      return this;
    }

    public ObjcVariablesExtension build() {

      ImmutableSet<VariableCategory> activeVariableCategories =
          activeVariableCategoriesBuilder.build();

      Preconditions.checkNotNull(ruleContext, "missing RuleContext");
      Preconditions.checkNotNull(buildConfiguration, "missing BuildConfigurationValue");
      Preconditions.checkNotNull(intermediateArtifacts, "missing IntermediateArtifacts");
      if (activeVariableCategories.contains(VariableCategory.EXECUTABLE_LINKING_VARIABLES)) {
        Preconditions.checkNotNull(frameworkSearchPaths, "missing FrameworkSearchPaths");
        Preconditions.checkNotNull(frameworkNames, "missing framework names");
        Preconditions.checkNotNull(weakFrameworkNames, "missing weak framework names");
        Preconditions.checkNotNull(libraryNames, "missing library names");
        Preconditions.checkNotNull(forceLoadArtifacts, "missing force-load artifacts");
        Preconditions.checkNotNull(depLinkopts, "missing dep linkopts");
        Preconditions.checkNotNull(attributeLinkopts, "missing attribute linkopts");
      }
      if (activeVariableCategories.contains(VariableCategory.DSYM_VARIABLES)) {
        Preconditions.checkNotNull(dsymSymbol, "missing dsym symbol artifact");
      }
      if (activeVariableCategories.contains(VariableCategory.LINKMAP_VARIABLES)) {
        Preconditions.checkNotNull(linkmap, "missing linkmap artifact");
      }

      return new ObjcVariablesExtension(
          ruleContext,
          intermediateArtifacts,
          buildConfiguration,
          frameworkSearchPaths,
          frameworkNames,
          weakFrameworkNames,
          libraryNames,
          forceLoadArtifacts,
          depLinkopts,
          attributeLinkopts,
          activeVariableCategories,
          dsymSymbol,
          linkmap,
          arcEnabled);
    }
  }
}
