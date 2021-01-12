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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINKOPT;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.shell.ShellUtils;
import java.util.Set;

/** Build variable extensions for templating a toolchain for objc builds. */
class ObjcVariablesExtension implements VariablesExtension {

  static final String PCH_FILE_VARIABLE_NAME = "pch_file";
  static final String FRAMEWORKS_PATH_NAME = "framework_paths";
  static final String MODULES_MAPS_DIR_NAME = "module_maps_dir";
  static final String OBJC_MODULE_CACHE_DIR_NAME = "_objc_module_cache";
  static final String OBJC_MODULE_CACHE_KEY = "modules_cache_path";
  static final String OBJ_LIST_PATH_VARIABLE_NAME = "obj_list_path";
  static final String ARCHIVE_PATH_VARIABLE_NAME = "archive_path";
  static final String FULLY_LINKED_ARCHIVE_PATH_VARIABLE_NAME = "fully_linked_archive_path";
  static final String OBJC_LIBRARY_EXEC_PATHS_VARIABLE_NAME = "objc_library_exec_paths";
  static final String CC_LIBRARY_EXEC_PATHS_VARIABLE_NAME = "cc_library_exec_paths";
  static final String IMPORTED_LIBRARY_EXEC_PATHS_VARIABLE_NAME = "imported_library_exec_paths";
  static final String LINKMAP_EXEC_PATH = "linkmap_exec_path";
  static final String BITCODE_SYMBOL_MAP_PATH_VARAIBLE_NAME = "bitcode_symbol_map_path";

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
  private final ObjcProvider objcProvider;
  private final CompilationArtifacts compilationArtifacts;
  private final Artifact fullyLinkArchive;
  private final IntermediateArtifacts intermediateArtifacts;
  private final BuildConfiguration buildConfiguration;
  private final ImmutableList<String> frameworkSearchPaths;
  private final Set<String> frameworkNames;
  private final ImmutableList<String> libraryNames;
  private final ImmutableSet<Artifact> forceLoadArtifacts;
  private final ImmutableList<String> attributeLinkopts;
  private final ImmutableSet<VariableCategory> activeVariableCategories;
  private final Artifact dsymSymbol;
  private final Artifact linkmap;
  private final Artifact bitcodeSymbolMap;
  private boolean arcEnabled = true;

  private ObjcVariablesExtension(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts,
      Artifact fullyLinkArchive,
      IntermediateArtifacts intermediateArtifacts,
      BuildConfiguration buildConfiguration,
      ImmutableList<String> frameworkSearchPaths,
      Set<String> frameworkNames,
      ImmutableList<String> libraryNames,
      ImmutableSet<Artifact> forceLoadArtifacts,
      ImmutableList<String> attributeLinkopts,
      ImmutableSet<VariableCategory> activeVariableCategories,
      Artifact dsymSymbol,
      Artifact linkmap,
      Artifact bitcodeSymbolMap,
      boolean arcEnabled) {
    this.ruleContext = ruleContext;
    this.objcProvider = objcProvider;
    this.compilationArtifacts = compilationArtifacts;
    this.fullyLinkArchive = fullyLinkArchive;
    this.intermediateArtifacts = intermediateArtifacts;
    this.buildConfiguration = buildConfiguration;
    this.frameworkSearchPaths = frameworkSearchPaths;
    this.frameworkNames = frameworkNames;
    this.libraryNames = libraryNames;
    this.forceLoadArtifacts = forceLoadArtifacts;
    this.attributeLinkopts = attributeLinkopts;
    this.activeVariableCategories = activeVariableCategories;
    this.dsymSymbol = dsymSymbol;
    this.linkmap = linkmap;
    this.bitcodeSymbolMap = bitcodeSymbolMap;
    this.arcEnabled = arcEnabled;
  }

  /** Type of build variable that can optionally exported by this extension. */
  public enum VariableCategory {
    ARCHIVE_VARIABLES,
    FULLY_LINK_VARIABLES,
    EXECUTABLE_LINKING_VARIABLES,
    DSYM_VARIABLES,
    LINKMAP_VARIABLES,
    BITCODE_VARIABLES
  }

  @Override
  public void addVariables(CcToolchainVariables.Builder builder) {
    addPchVariables(builder);
    addModuleMapVariables(builder);
    if (activeVariableCategories.contains(VariableCategory.ARCHIVE_VARIABLES)) {
      addArchiveVariables(builder);
    }
    if (activeVariableCategories.contains(VariableCategory.FULLY_LINK_VARIABLES)) {
      addFullyLinkArchiveVariables(builder);
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
    if (activeVariableCategories.contains(VariableCategory.BITCODE_VARIABLES)) {
      addBitcodeVariables(builder);
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
        MODULES_MAPS_DIR_NAME,
        intermediateArtifacts
            .moduleMap()
            .getArtifact()
            .getExecPath()
            .getParentDirectory()
            .toString());
    builder.addStringVariable(
        OBJC_MODULE_CACHE_KEY,
        buildConfiguration.getGenfilesFragment(ruleContext.getRepository())
            + "/"
            + OBJC_MODULE_CACHE_DIR_NAME);
  }

  private void addArchiveVariables(CcToolchainVariables.Builder builder) {
    builder.addStringVariable(
        OBJ_LIST_PATH_VARIABLE_NAME,
        intermediateArtifacts.archiveObjList().getExecPathString());
    builder.addStringVariable(
        ARCHIVE_PATH_VARIABLE_NAME, compilationArtifacts.getArchive().get().getExecPathString());
  }

  private void addFullyLinkArchiveVariables(CcToolchainVariables.Builder builder) {
    builder.addStringVariable(
        FULLY_LINKED_ARCHIVE_PATH_VARIABLE_NAME, fullyLinkArchive.getExecPathString());

    // ObjcProvider.getObjcLibraries contains both libraries from objc providers
    // as well as those from CcInfo. ObjcProvider.getCcLibraries only contains
    // those from CcInfo. We have to split these lists to make sure duplicate
    // libraries are not included in the fully linked archive.
    ImmutableSet<Artifact> ccLibs = ImmutableSet.copyOf(objcProvider.getCcLibraries());
    Predicate<Artifact> isNotCcLib = library -> !ccLibs.contains(library);
    Iterable<Artifact> objcLibraries =
        objcProvider.getObjcLibraries().stream().filter(isNotCcLib).collect(toImmutableList());

    builder.addStringSequenceVariable(
        OBJC_LIBRARY_EXEC_PATHS_VARIABLE_NAME, Artifact.toExecPaths(objcLibraries));
    builder.addStringSequenceVariable(
        CC_LIBRARY_EXEC_PATHS_VARIABLE_NAME,
        Artifact.toExecPaths(objcProvider.getCcLibraries()));
    builder.addStringSequenceVariable(
        IMPORTED_LIBRARY_EXEC_PATHS_VARIABLE_NAME,
        Artifact.toExecPaths(objcProvider.get(IMPORTED_LIBRARY).toList()));
  }

  private void addExecutableLinkVariables(CcToolchainVariables.Builder builder) {
    builder.addStringSequenceVariable(FRAMEWORKS_PATH_NAME, frameworkSearchPaths);
    builder.addStringSequenceVariable(
        FRAMEWORK_NAMES_VARIABLE_NAME, frameworkNames);
    builder.addStringSequenceVariable(
        WEAK_FRAMEWORK_NAMES_VARIABLE_NAME,
            SdkFramework.names(objcProvider.get(ObjcProvider.WEAK_SDK_FRAMEWORK)));
    builder.addStringSequenceVariable(LIBRARY_NAMES_VARIABLE_NAME, libraryNames);
    builder.addStringVariable(
        FILELIST_VARIABLE_NAME, intermediateArtifacts.linkerObjList().getExecPathString());
    builder.addStringVariable(
        LINKED_BINARY_VARIABLE_NAME,
        ruleContext.getFragment(ObjcConfiguration.class).shouldStripBinary()
            ? intermediateArtifacts.unstrippedSingleArchitectureBinary().getExecPathString()
            : intermediateArtifacts.strippedSingleArchitectureBinary().getExecPathString());

    builder.addStringSequenceVariable(
        FORCE_LOAD_EXEC_PATHS_VARIABLE_NAME,
        Artifact.toExecPaths(forceLoadArtifacts));
    builder.addStringSequenceVariable(DEP_LINKOPTS_VARIABLE_NAME, objcProvider.get(LINKOPT));
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

  private void addBitcodeVariables(CcToolchainVariables.Builder builder) {
    builder.addStringVariable(
        BITCODE_SYMBOL_MAP_PATH_VARAIBLE_NAME, bitcodeSymbolMap.getExecPathString());
  }

  /** A Builder for {@link ObjcVariablesExtension}. */
  static class Builder {
    private RuleContext ruleContext;
    private ObjcProvider objcProvider;
    private CompilationArtifacts compilationArtifacts;
    private Artifact fullyLinkArchive;
    private IntermediateArtifacts intermediateArtifacts;
    private BuildConfiguration buildConfiguration;
    private ImmutableList<String> frameworkSearchPaths;
    private Set<String> frameworkNames;
    private ImmutableSet<Artifact> forceLoadArtifacts;
    private ImmutableList<String> libraryNames;
    private ImmutableList<String> attributeLinkopts;
    private Artifact dsymSymbol;
    private Artifact linkmap;
    private Artifact bitcodeSymbolMap;
    private boolean arcEnabled = true;

    private final ImmutableSet.Builder<VariableCategory> activeVariableCategoriesBuilder =
        ImmutableSet.builder();

    /** Sets the {@link RuleContext} for this extension. */
    public Builder setRuleContext(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
      return this;
    }

    /** Sets the {@link ObjcProvider} for this extension. */
    public Builder setObjcProvider(ObjcProvider objcProvider) {
      this.objcProvider = Preconditions.checkNotNull(objcProvider);
      return this;
    }

    /** Sets the {@link CompilationArtifacts} for this extension. */
    public Builder setCompilationArtifacts(CompilationArtifacts compilationArtifacts) {
      this.compilationArtifacts = Preconditions.checkNotNull(compilationArtifacts);
      return this;
    }

    /** Sets the output of the fully link action. */
    public Builder setFullyLinkArchive(Artifact fullyLinkArchive) {
      this.fullyLinkArchive = Preconditions.checkNotNull(fullyLinkArchive);
      return this;
    }

    /** Sets the {@link IntermediateArtifacts} for this extension. */
    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = Preconditions.checkNotNull(intermediateArtifacts);
      return this;
    }

    /** Sets the configuration for this extension. */
    public Builder setConfiguration(BuildConfiguration buildConfiguration) {
      this.buildConfiguration = Preconditions.checkNotNull(buildConfiguration);
      return this;
    }

    /** Sets the framework search paths to be passed to the compiler/linker using {@code -F}. */
    public Builder setFrameworkSearchPath(ImmutableList<String> frameworkSearchPaths) {
      this.frameworkSearchPaths = Preconditions.checkNotNull(frameworkSearchPaths);
      return this;
    }

    /** Sets the framework names to be passed to the linker using {@code -framework}. */
    public Builder setFrameworkNames(Set<String> frameworkNames) {
      this.frameworkNames = Preconditions.checkNotNull(frameworkNames);
      return this;
    }

    /** Sets binary input files to be passed to the linker with "-l" flags. */
    public Builder setLibraryNames(ImmutableList<String> libraryNames) {
      this.libraryNames = Preconditions.checkNotNull(libraryNames);
      return this;
    }

    /** Sets artifacts to be passed to the linker with {@code -force_load}. */
    public Builder setForceLoadArtifacts(ImmutableSet<Artifact> forceLoadArtifacts) {
      this.forceLoadArtifacts = Preconditions.checkNotNull(forceLoadArtifacts);
      return this;
    }

    /** Sets linkopts arising from rule attributes. */
    public Builder setAttributeLinkopts(ImmutableList<String> attributeLinkopts) {
      this.attributeLinkopts = Preconditions.checkNotNull(attributeLinkopts);
      return this;
    }

    /** Sets the given {@link VariableCategory} as active for this extension. */
    public Builder addVariableCategory(VariableCategory variableCategory) {
      this.activeVariableCategoriesBuilder.add(Preconditions.checkNotNull(variableCategory));
      return this;
    }

    /** Sets the Artifact for the dsym symbol file. */
    public Builder setDsymSymbol(Artifact dsymSymbol) {
      this.dsymSymbol = dsymSymbol;
      return this;
    }

    /** Sets the Artifact for the linkmap. */
    public Builder setLinkmap(Artifact linkmap) {
      this.linkmap = linkmap;
      return this;
    }

    /** Sets the Artifact for the bitcode symbol map. */
    public Builder setBitcodeSymbolMap(Artifact bitcodeSymbolMap) {
      this.bitcodeSymbolMap = bitcodeSymbolMap;
      return this;
    }

    /** Sets whether ARC is enabled. */
    public Builder setArcEnabled(boolean enabled) {
      this.arcEnabled = enabled;
      return this;
    }

    public ObjcVariablesExtension build() {

      ImmutableSet<VariableCategory> activeVariableCategories =
          activeVariableCategoriesBuilder.build();

      Preconditions.checkNotNull(ruleContext, "missing RuleContext");
      Preconditions.checkNotNull(buildConfiguration, "missing BuildConfiguration");
      Preconditions.checkNotNull(intermediateArtifacts, "missing IntermediateArtifacts");
      if (activeVariableCategories.contains(VariableCategory.ARCHIVE_VARIABLES)) {
        Preconditions.checkNotNull(compilationArtifacts, "missing CompilationArtifacts");
      }
      if (activeVariableCategories.contains(VariableCategory.FULLY_LINK_VARIABLES)) {
        Preconditions.checkNotNull(objcProvider, "missing ObjcProvider");
        Preconditions.checkNotNull(fullyLinkArchive, "missing fully-link archive");
      }
      if (activeVariableCategories.contains(VariableCategory.EXECUTABLE_LINKING_VARIABLES)) {
        Preconditions.checkNotNull(objcProvider, "missing ObjcProvider");
        Preconditions.checkNotNull(frameworkSearchPaths, "missing FrameworkSearchPaths");
        Preconditions.checkNotNull(frameworkNames, "missing framework names");
        Preconditions.checkNotNull(libraryNames, "missing library names");
        Preconditions.checkNotNull(forceLoadArtifacts, "missing force-load artifacts");
        Preconditions.checkNotNull(attributeLinkopts, "missing attribute linkopts");
      }
      if (activeVariableCategories.contains(VariableCategory.DSYM_VARIABLES)) {
        Preconditions.checkNotNull(dsymSymbol, "missing dsym symbol artifact");
      }
      if (activeVariableCategories.contains(VariableCategory.LINKMAP_VARIABLES)) {
        Preconditions.checkNotNull(linkmap, "missing linkmap artifact");
      }
      if (activeVariableCategories.contains(VariableCategory.BITCODE_VARIABLES)) {
        Preconditions.checkNotNull(bitcodeSymbolMap, "missing bitcode symbol map artifact");
      }

      return new ObjcVariablesExtension(
          ruleContext,
          objcProvider,
          compilationArtifacts,
          fullyLinkArchive,
          intermediateArtifacts,
          buildConfiguration,
          frameworkSearchPaths,
          frameworkNames,
          libraryNames,
          forceLoadArtifacts,
          attributeLinkopts,
          activeVariableCategories,
          dsymSymbol,
          linkmap,
          bitcodeSymbolMap,
          arcEnabled);
    }
  }
}
