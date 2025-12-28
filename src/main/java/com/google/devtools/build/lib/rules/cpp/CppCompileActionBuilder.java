// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

/** Builder class to construct C++ compile actions. */
public final class CppCompileActionBuilder implements StarlarkValue {

  private final ActionOwner owner;
  private boolean shareable;
  private final BuildConfigurationValue configuration;
  private CcToolchainFeatures.FeatureConfiguration featureConfiguration;
  private CcToolchainVariables variables = CcToolchainVariables.empty();
  private Artifact sourceFile;
  private final NestedSetBuilder<Artifact> mandatoryInputsBuilder;
  private Artifact outputFile;
  private Artifact modmapFile;
  private Artifact modmapInputFile;
  private NestedSet<Artifact> moduleFiles;
  private Artifact dwoFile;
  private Artifact ltoIndexingFile;
  private Artifact dotdFile;
  private Artifact diagnosticsFile;
  private Artifact gcnoFile;
  private CcCompilationContext ccCompilationContext = null;
  private final List<String> pluginOpts = new ArrayList<>();
  private CoptsFilter coptsFilter = CoptsFilter.alwaysPasses();
  private ImmutableList<PathFragment> extraSystemIncludePrefixes = ImmutableList.of();
  private boolean usePic;
  private final CppConfiguration cppConfiguration;
  private final ArrayList<Artifact> additionalIncludeScanningRoots;
  private Boolean shouldScanIncludes;
  private Map<String, String> executionInfo = new LinkedHashMap<>();
  private final CcToolchainProvider ccToolchain;
  @Nullable private String actionName;
  private ImmutableList<Artifact> buildInfoHeaderArtifacts = ImmutableList.of();
  private NestedSet<Artifact> cacheKeyInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private NestedSet<Artifact> additionalPrunableHeaders =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private ImmutableList<Artifact> additionalOutputs = ImmutableList.of();
  private boolean needsIncludeValidation;

  // New fields need to be added to the copy constructor.

  /** Creates a builder from an owner and a configuration. */
  public CppCompileActionBuilder(
      ActionOwner owner, CcToolchainProvider ccToolchain, BuildConfigurationValue configuration) {

    this.owner = owner;
    this.shareable = false;
    this.configuration = configuration;
    this.cppConfiguration = configuration.getFragment(CppConfiguration.class);
    this.mandatoryInputsBuilder = NestedSetBuilder.stableOrder();
    this.additionalIncludeScanningRoots = new ArrayList<>();
    this.ccToolchain = ccToolchain;
  }

  /** Creates a builder from a rule and a configuration. */
  public CppCompileActionBuilder(
      ActionConstructionContext actionConstructionContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      String cppToolchainType) {

    this(getActionOwner(actionConstructionContext, cppToolchainType), ccToolchain, configuration);
  }

  /** Creates a builder that is a copy of another builder. */
  public CppCompileActionBuilder(CppCompileActionBuilder other) {
    this.owner = other.owner;
    this.shareable = other.shareable;
    this.featureConfiguration = other.featureConfiguration;
    this.sourceFile = other.sourceFile;
    this.mandatoryInputsBuilder =
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(other.mandatoryInputsBuilder.build());
    this.additionalIncludeScanningRoots = new ArrayList<>();
    this.additionalIncludeScanningRoots.addAll(other.additionalIncludeScanningRoots);
    this.outputFile = other.outputFile;
    this.dwoFile = other.dwoFile;
    this.ltoIndexingFile = other.ltoIndexingFile;
    this.dotdFile = other.dotdFile;
    this.gcnoFile = other.gcnoFile;
    this.ccCompilationContext = other.ccCompilationContext;
    this.pluginOpts.addAll(other.pluginOpts);
    this.coptsFilter = other.coptsFilter;
    this.extraSystemIncludePrefixes = other.extraSystemIncludePrefixes;
    this.cppConfiguration = other.cppConfiguration;
    this.configuration = other.configuration;
    this.usePic = other.usePic;
    this.shouldScanIncludes = other.shouldScanIncludes;
    this.executionInfo = new LinkedHashMap<>(other.executionInfo);
    this.ccToolchain = other.ccToolchain;
    this.actionName = other.actionName;
    this.additionalOutputs = other.additionalOutputs;
    this.needsIncludeValidation = other.needsIncludeValidation;
    this.moduleFiles = other.moduleFiles;
    this.modmapFile = other.modmapFile;
    this.modmapInputFile = other.modmapInputFile;
  }

  static ActionOwner getActionOwner(
      ActionConstructionContext actionConstructionContext, String cppToolchainType) {
    ActionOwner actionOwner = null;
    if (actionConstructionContext instanceof RuleContext ruleContext
        && ruleContext.useAutoExecGroups()) {
      actionOwner =
          actionConstructionContext.getActionOwner(
              Label.parseCanonicalUnchecked(cppToolchainType).toString());
    }
    return actionOwner == null ? actionConstructionContext.getActionOwner() : actionOwner;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setSourceFile(Artifact sourceFile) {
    Preconditions.checkState(
        this.sourceFile == null,
        "New source file %s trying to overwrite old source file %s",
        sourceFile,
        this.sourceFile);
    return setSourceFileUnchecked(sourceFile);
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setSourceFile(Artifact.TreeFileArtifact sourceFile) {
    Preconditions.checkState(
        !(this.sourceFile instanceof Artifact.TreeFileArtifact),
        "New source file %s trying to overwrite old source file %s also a tree file artifact",
        sourceFile,
        this.sourceFile);
    return setSourceFileUnchecked(sourceFile);
  }

  @CanIgnoreReturnValue
  private CppCompileActionBuilder setSourceFileUnchecked(Artifact sourceFile) {
    this.sourceFile = sourceFile;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setAdditionalOutputs(ImmutableList<Artifact> additionalOutputs) {
    this.additionalOutputs = additionalOutputs;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setNeedsIncludeValidation(boolean needsIncludeValidation) {
    this.needsIncludeValidation = needsIncludeValidation;
    return this;
  }

  public ActionOwner getOwner() {
    return owner;
  }

  public Artifact getSourceFile() {
    return sourceFile;
  }

  public CcCompilationContext getCcCompilationContext() {
    return ccCompilationContext;
  }

  public String getActionName() {
    if (actionName != null) {
      return actionName;
    }
    PathFragment sourcePath = sourceFile.getExecPath();
    if (CppFileTypes.CPP_MODULE_MAP.matches(sourcePath)) {
      return CppActionNames.CPP_MODULE_COMPILE;
    } else if (CppFileTypes.CPP_HEADER.matches(sourcePath)) {
      // TODO(bazel-team): Handle C headers that probably don't work in C++ mode.
      if (featureConfiguration.isEnabled(CppRuleClasses.PARSE_HEADERS)) {
        return CppActionNames.CPP_HEADER_PARSING;
      }
      // CcCommon.collectCAndCppSources() ensures we do not add headers to
      // the compilation artifacts unless 'parse_headers' is set.
      throw new IllegalStateException();
    } else if (CppFileTypes.C_SOURCE.matches(sourcePath)) {
      return CppActionNames.C_COMPILE;
    } else if (CppFileTypes.CPP_SOURCE.matches(sourcePath)) {
      return CppActionNames.CPP_COMPILE;
    } else if (CppFileTypes.OBJC_SOURCE.matches(sourcePath)) {
      return CppActionNames.OBJC_COMPILE;
    } else if (CppFileTypes.OBJCPP_SOURCE.matches(sourcePath)) {
      return CppActionNames.OBJCPP_COMPILE;
    } else if (CppFileTypes.ASSEMBLER.matches(sourcePath)) {
      return CppActionNames.ASSEMBLE;
    } else if (CppFileTypes.ASSEMBLER_WITH_C_PREPROCESSOR.matches(sourcePath)) {
      return CppActionNames.PREPROCESS_ASSEMBLE;
    } else if (CppFileTypes.CLIF_INPUT_PROTO.matches(sourcePath)) {
      return CppActionNames.CLIF_MATCH;
    } else if (CppFileTypes.CPP_MODULE.matches(sourcePath)) {
      return CppActionNames.CPP_MODULE_CODEGEN;
    }
    // CcCompilationHelper ensures CppCompileAction only gets instantiated for supported file types.
    throw new IllegalStateException();
  }

  /**
   * Builds the Action as configured and performs some validations on the action. Throws {@link
   * IllegalStateException} to report errors. Prefer {@link
   * CppCompileActionBuilder#buildOrThrowRuleError(RuleErrorConsumer)} over this method whenever
   * possible (meaning whenever you have access to {@link RuleContext}).
   *
   * <p>This method may be called multiple times to create multiple compile actions (usually after
   * calling some setters to modify the generated action).
   */
  public CppCompileAction buildOrThrowIllegalStateException() {
    try {
      return buildAndVerify();
    } catch (UnconfiguredActionConfigException | EvalException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Exception thrown when the action is not configured in the toolchain. */
  public static final class UnconfiguredActionConfigException extends Exception {
    private UnconfiguredActionConfigException(String actionName) {
      super(String.format("Expected action_config for '%s' to be configured", actionName));
    }
  }

  /**
   * Builds the Action as configured and performs some validations on the action. Uses given {@link
   * Consumer} to collect validation errors.
   */
  public CppCompileAction buildAndVerify() throws UnconfiguredActionConfigException, EvalException {
    Preconditions.checkNotNull(featureConfiguration);
    boolean useHeaderModules = useHeaderModules();

    String actionName = getActionName();
    if (featureConfiguration.actionIsConfigured(actionName)) {
      for (String executionRequirement :
          featureConfiguration.getToolRequirementsForAction(actionName)) {
        executionInfo.put(executionRequirement, "");
      }
    } else {
      throw new UnconfiguredActionConfigException(actionName);
    }

    // If include scanning is enabled, we can use the filegroup without header files - they are
    // found by include scanning.  We still need the system framework headers since they are not
    // being scanned right now, but those are placed in the "compile" file group.
    // TODO(djasper): When not include scanning, getCompilerFiles() should be enough here, but that
    // doesn't currently include GRTE headers.
    NestedSet<Artifact> compilerFilesWithoutIncludes =
        ccToolchain.getCompilerFilesWithoutIncludes();
    if (compilerFilesWithoutIncludes.isEmpty()) {
      compilerFilesWithoutIncludes = ccToolchain.getCompilerFiles();
    }
    addTransitiveMandatoryInputs(
        getShouldScanIncludes()
            ? compilerFilesWithoutIncludes
            : configuration.getFragment(CppConfiguration.class).useSpecificToolFiles()
                    && !getSourceFile().isTreeArtifact()
                ? (getActionName().equals(CppActionNames.ASSEMBLE)
                    ? ccToolchain.getAsFiles()
                    : ccToolchain.getCompilerFiles())
                : ccToolchain.getAllFiles());

    NestedSet<Artifact> realMandatorySpawnInputs = buildMandatoryInputs();
    NestedSet<Artifact> realMandatoryInputs =
        NestedSet.<Artifact>builder(Order.STABLE_ORDER)
            .addTransitive(realMandatorySpawnInputs)
            .addTransitive(cacheKeyInputs)
            .build();
    NestedSet<Artifact> prunableHeaders = additionalPrunableHeaders;

    configuration.modifyExecutionInfo(
        executionInfo,
        CppCompileAction.actionNameToMnemonic(
            actionName, featureConfiguration, cppConfiguration.useCppCompileHeaderMnemonic()));

    // Copying the collections is needed to make the builder reusable.
    return new CppCompileAction(
        owner,
        featureConfiguration,
        variables,
        sourceFile,
        configuration,
        shareable,
        shouldScanIncludes,
        usePic,
        useHeaderModules,
        realMandatoryInputs,
        realMandatorySpawnInputs,
        getBuiltinIncludeFiles(),
        prunableHeaders,
        outputFile,
        dotdFile,
        diagnosticsFile,
        gcnoFile,
        dwoFile,
        ltoIndexingFile,
        ccCompilationContext,
        coptsFilter,
        ImmutableList.copyOf(additionalIncludeScanningRoots),
        ImmutableMap.copyOf(executionInfo),
        actionName,
        needsIncludeValidation,
        getBuiltinIncludeDirectories(),
        ccToolchain.getGrepIncludes(),
        additionalOutputs,
        moduleFiles,
        modmapInputFile);
  }

  private ImmutableList<Artifact> getBuiltinIncludeFiles() throws EvalException {
    ImmutableList<Artifact> builtinIncludeFiles = ccToolchain.getBuiltinIncludeFiles();
    if (buildInfoHeaderArtifacts.isEmpty()) {
      return builtinIncludeFiles;
    }
    if (builtinIncludeFiles.isEmpty()) {
      return buildInfoHeaderArtifacts;
    }
    return ImmutableList.<Artifact>builderWithExpectedSize(
            builtinIncludeFiles.size() + buildInfoHeaderArtifacts.size())
        .addAll(builtinIncludeFiles)
        .addAll(buildInfoHeaderArtifacts)
        .build();
  }

  private boolean shouldParseShowIncludes() {
    return featureConfiguration.isEnabled(CppRuleClasses.PARSE_SHOWINCLUDES);
  }

  /** Returns the list of mandatory inputs for the {@link CppCompileAction} as configured. */
  NestedSet<Artifact> buildMandatoryInputs() throws EvalException {
    Preconditions.checkNotNull(ccCompilationContext);
    NestedSetBuilder<Artifact> realMandatoryInputsBuilder = NestedSetBuilder.compileOrder();
    realMandatoryInputsBuilder.addTransitive(mandatoryInputsBuilder.build());
    realMandatoryInputsBuilder.addAll(getBuiltinIncludeFiles());
    if (useHeaderModules() && !getShouldScanIncludes()) {
      realMandatoryInputsBuilder.addTransitive(ccCompilationContext.getTransitiveModules(usePic));
    }
    ccCompilationContext.addAdditionalInputs(realMandatoryInputsBuilder);
    realMandatoryInputsBuilder.add(Preconditions.checkNotNull(sourceFile));
    if (ccToolchain.getGrepIncludes() != null) {
      realMandatoryInputsBuilder.add(ccToolchain.getGrepIncludes());
    }
    if (!getShouldScanIncludes() && dotdFile == null && !shouldParseShowIncludes()) {
      realMandatoryInputsBuilder.addTransitive(ccCompilationContext.getDeclaredIncludeSrcs());
      realMandatoryInputsBuilder.addTransitive(additionalPrunableHeaders);
    }
    if (modmapFile != null) {
      realMandatoryInputsBuilder.add(modmapFile).add(Preconditions.checkNotNull(modmapInputFile));
    }
    return realMandatoryInputsBuilder.build();
  }

  NestedSet<Artifact> getPrunableHeaders() {
    return additionalPrunableHeaders;
  }

  NestedSet<Artifact> getInputsForInvalidation() {
    return ccCompilationContext.getDeclaredIncludeSrcs();
  }

  private boolean useHeaderModules(Artifact sourceFile) {
    Preconditions.checkNotNull(featureConfiguration);
    Preconditions.checkNotNull(sourceFile);
    return featureConfiguration.isEnabled(CppRuleClasses.USE_HEADER_MODULES)
        && (sourceFile.isFileType(CppFileTypes.CPP_SOURCE)
            || sourceFile.isFileType(CppFileTypes.CPP_HEADER)
            || sourceFile.isFileType(CppFileTypes.CPP_MODULE_MAP));
  }

  private boolean useHeaderModules() {
    return useHeaderModules(sourceFile);
  }

  /**
   * Set action name that is used to pick the right action_config and features from {@link
   * FeatureConfiguration}. By default the action name is decided from the source filetype.
   */
  @CanIgnoreReturnValue
  public CppCompileActionBuilder setActionName(String actionName) {
    Preconditions.checkState(
        this.actionName == null,
        "New actionName %s trying to overwrite old name %s",
        actionName,
        this.actionName);
    this.actionName = actionName;
    return this;
  }

  /** Sets the feature configuration to be used for the action. */
  @CanIgnoreReturnValue
  public CppCompileActionBuilder setFeatureConfiguration(
      FeatureConfiguration featureConfiguration) {
    Preconditions.checkNotNull(featureConfiguration);
    this.featureConfiguration = featureConfiguration;
    return this;
  }

  FeatureConfiguration getFeatureConfiguration() {
    return featureConfiguration;
  }

  /** Sets the feature build variables to be used for the action. */
  @CanIgnoreReturnValue
  public CppCompileActionBuilder setVariables(CcToolchainVariables variables) {
    this.variables = variables;
    return this;
  }

  /** Returns the build variables to be used for the action. */
  public CcToolchainVariables getVariables() {
    return variables;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder addExecutionInfo(Map<String, String> executionInfo) {
    this.executionInfo.putAll(executionInfo);
    return this;
  }

  Map<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder addMandatoryInputs(NestedSet<Artifact> artifacts) {
    mandatoryInputsBuilder.addTransitive(artifacts);
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder addMandatoryInputs(List<Artifact> artifacts) {
    mandatoryInputsBuilder.addAll(artifacts);
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder addTransitiveMandatoryInputs(NestedSet<Artifact> artifacts) {
    mandatoryInputsBuilder.addTransitive(artifacts);
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder addAdditionalIncludeScanningRoots(
      List<Artifact> additionalIncludeScanningRoots) {
    this.additionalIncludeScanningRoots.addAll(additionalIncludeScanningRoots);
    return this;
  }

  public boolean useDotdFile(Artifact sourceFile) {
    return CppFileTypes.headerDiscoveryRequired(sourceFile) && !useHeaderModules(sourceFile);
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setOutputs(
      Artifact outputFile, Artifact dotdFile, Artifact diagnosticsFile) {
    this.outputFile = outputFile;
    this.dotdFile = dotdFile;
    this.diagnosticsFile = diagnosticsFile;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setDwoFile(Artifact dwoFile) {
    this.dwoFile = dwoFile;
    return this;
  }

  /**
   * Set the minimized bitcode file emitted by this (ThinLTO) compilation that can be used in place
   * of the full bitcode outputFile in the LTO indexing step.
   */
  @CanIgnoreReturnValue
  public CppCompileActionBuilder setLtoIndexingFile(Artifact ltoIndexingFile) {
    this.ltoIndexingFile = ltoIndexingFile;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setGcnoFile(Artifact gcnoFile) {
    this.gcnoFile = gcnoFile;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setCcCompilationContext(
      CcCompilationContext ccCompilationContext) {
    this.ccCompilationContext = ccCompilationContext;
    return this;
  }

  /** Sets whether the CompileAction should use pic mode. */
  @CanIgnoreReturnValue
  public CppCompileActionBuilder setPicMode(boolean usePic) {
    this.usePic = usePic;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setShareable(boolean shareable) {
    this.shareable = shareable;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setShouldScanIncludes(boolean shouldScanIncludes) {
    this.shouldScanIncludes = shouldScanIncludes;
    return this;
  }

  public boolean getShouldScanIncludes() {
    if (shouldScanIncludes != null) {
      return shouldScanIncludes;
    }

    boolean moduleDepsScanningAction =
        actionName != null && actionName.equals(CppActionNames.CPP_MODULE_DEPS_SCANNING);

    // Three things have to be true to perform #include scanning:
    //  1. The toolchain configuration has to generally enable it.
    //     This is the default unless the --nocc_include_scanning flag was specified.
    //  2. The action is not CPP_MODULE_DEPS_SCANNING.
    //  3. The rule or package must not disable it via 'features = ["-cc_include_scanning"]'.
    //     Normally the scanner is enabled, but rules with precisely specified
    //     dependencies not understood by the scanner can selectively disable it.
    //  4. The file must not be a not-for-preprocessing assembler source file.
    //     Assembler without C preprocessing can use the '.include' pseudo-op which is not
    //     understood by the include scanner, so we'll disable scanning, and instead require
    //     the declared sources to state (possibly overapproximate) the dependencies.
    //     Assembler with preprocessing can also use '.include', but supporting both kinds
    //     of inclusion for that use-case is ridiculous.
    shouldScanIncludes =
        configuration.getFragment(CppConfiguration.class).shouldScanIncludes()
            && featureConfiguration.getRequestedFeatures().contains("cc_include_scanning")
            && !moduleDepsScanningAction
            && !sourceFile.isFileType(CppFileTypes.ASSEMBLER)
            && !sourceFile.isFileType(CppFileTypes.CPP_MODULE);
    return shouldScanIncludes;
  }

  public CcToolchainProvider getToolchain() {
    return ccToolchain;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setCoptsFilter(CoptsFilter coptsFilter) {
    this.coptsFilter = Preconditions.checkNotNull(coptsFilter);
    return this;
  }

  CoptsFilter getCoptsFilter() {
    return coptsFilter;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setBuildInfoHeaderArtifacts(
      ImmutableList<Artifact> buildInfoHeaderArtifacts) {
    this.buildInfoHeaderArtifacts = buildInfoHeaderArtifacts;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setCacheKeyInputs(NestedSet<Artifact> cacheKeyInputs) {
    this.cacheKeyInputs = cacheKeyInputs;
    return this;
  }

  ActionEnvironment getActionEnvironment() {
    return configuration.getActionEnvironment();
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setAdditionalPrunableHeaders(
      NestedSet<Artifact> additionalPrunableHeaders) {
    this.additionalPrunableHeaders = Preconditions.checkNotNull(additionalPrunableHeaders);
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setModmapFile(Artifact modmapFile) {
    this.modmapFile = modmapFile;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setModmapInputFile(Artifact modmapInputFile) {
    this.modmapInputFile = modmapInputFile;
    return this;
  }

  @CanIgnoreReturnValue
  public CppCompileActionBuilder setModuleFiles(NestedSet<Artifact> moduleFiles) {
    this.moduleFiles = moduleFiles;
    return this;
  }

  ImmutableList<PathFragment> getBuiltinIncludeDirectories() throws EvalException {
    return ccToolchain.getBuiltInIncludeDirectories();
  }

  public boolean shouldCompileHeaders() {
    Preconditions.checkNotNull(featureConfiguration);
    return CcToolchainProvider.shouldProcessHeaders(featureConfiguration, cppConfiguration);
  }
}
