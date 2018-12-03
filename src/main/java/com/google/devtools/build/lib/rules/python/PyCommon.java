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
package com.google.devtools.build.lib.rules.python;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.PythonInfo;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.Util;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.LocalMetadataCollector;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.GeneratedMessage.GeneratedExtension;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * A helper class for Python rules.
 */
public final class PyCommon {

  public static final String PYTHON_SKYLARK_PROVIDER_NAME = "py";
  public static final String TRANSITIVE_PYTHON_SRCS = "transitive_sources";
  public static final String IS_USING_SHARED_LIBRARY = "uses_shared_libraries";
  public static final String IMPORTS = "imports";

  private static final LocalMetadataCollector METADATA_COLLECTOR = new LocalMetadataCollector() {
    @Override
    public void collectMetadataArtifacts(Iterable<Artifact> artifacts,
        AnalysisEnvironment analysisEnvironment, NestedSetBuilder<Artifact> metadataFilesBuilder) {
      // Python doesn't do any compilation, so we simply return the empty set.
    }
  };

  private final RuleContext ruleContext;

  private Artifact executable = null;

  private NestedSet<Artifact> transitivePythonSources;

  private PythonVersion sourcesVersion;
  private PythonVersion version = null;
  private Map<PathFragment, Artifact> convertedFiles;

  private NestedSet<Artifact> filesToBuild = null;

  public PyCommon(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  public void initCommon(PythonVersion defaultVersion) {
    this.sourcesVersion = getSrcsVersionAttr(ruleContext);
    this.version = ruleContext.getFragment(PythonConfiguration.class)
        .getPythonVersion(defaultVersion);
    this.transitivePythonSources = collectTransitivePythonSources();
    checkSourceIsCompatible(this.version, this.sourcesVersion, ruleContext.getLabel());
  }

  public PythonVersion getVersion() {
    return version;
  }

  public void initBinary(List<Artifact> srcs) {
    Preconditions.checkNotNull(version);

    validatePackageName();
    if (OS.getCurrent() == OS.WINDOWS) {
      executable =
          ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".exe");
    } else {
      executable = ruleContext.createOutputArtifact();
    }
    if (this.version == PythonVersion.PY2AND3) {
      // TODO(bazel-team): we need to create two actions
      ruleContext.ruleError("PY2AND3 is not yet implemented");
    }

    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().addAll(srcs).add(executable);

    if (ruleContext.getFragment(PythonConfiguration.class).buildPythonZip()) {
      filesToBuildBuilder.add(getPythonZipArtifact(executable));
    } else if (OS.getCurrent() == OS.WINDOWS) {
      // TODO(bazel-team): Here we should check target platform instead of using OS.getCurrent().
      // On Windows, add the python stub launcher in the set of files to build.
      filesToBuildBuilder.add(getPythonLauncherArtifact(executable));
    }

    filesToBuild = filesToBuildBuilder.build();

    if (ruleContext.hasErrors()) {
      return;
    }

    addPyExtraActionPseudoAction();
  }

  /** @return An artifact next to the executable file with ".zip" suffix */
  public Artifact getPythonZipArtifact(Artifact executable) {
    return ruleContext.getRelatedArtifact(executable.getRootRelativePath(), ".zip");
  }

  /** @return An artifact next to the executable file with no suffix, only used on Windows */
  public Artifact getPythonLauncherArtifact(Artifact executable) {
    return ruleContext.getRelatedArtifact(executable.getRootRelativePath(), "");
  }

  public void addCommonTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder,
      PythonSemantics semantics,
      NestedSet<Artifact> filesToBuild,
      NestedSet<String> imports) {

    builder
        .addNativeDeclaredProvider(
            InstrumentedFilesCollector.collect(
                ruleContext,
                semantics.getCoverageInstrumentationSpec(),
                METADATA_COLLECTOR,
                filesToBuild,
                /* reportedToActualSources= */ NestedSetBuilder.create(Order.STABLE_ORDER)))
        .addSkylarkTransitiveInfo(
            PYTHON_SKYLARK_PROVIDER_NAME,
            createSourceProvider(this.transitivePythonSources, usesSharedLibraries(), imports))
        // Python targets are not really compilable. The best we can do is make sure that all
        // generated source files are ready.
        .addOutputGroup(OutputGroupInfo.FILES_TO_COMPILE, transitivePythonSources)
        .addOutputGroup(OutputGroupInfo.COMPILATION_PREREQUISITES, transitivePythonSources);
  }

  /**
   * Returns a Skylark struct for exposing transitive Python sources:
   *
   * <p>addSkylarkTransitiveInfo(PYTHON_SKYLARK_PROVIDER_NAME, createSourceProvider(...))
   */
  public static StructImpl createSourceProvider(
      NestedSet<Artifact> transitivePythonSources,
      boolean isUsingSharedLibrary,
      NestedSet<String> imports) {
    return StructProvider.STRUCT.create(
        ImmutableMap.<String, Object>of(
            TRANSITIVE_PYTHON_SRCS,
            SkylarkNestedSet.of(Artifact.class, transitivePythonSources),
            IS_USING_SHARED_LIBRARY,
            isUsingSharedLibrary,
            IMPORTS,
            SkylarkNestedSet.of(String.class, imports)),
        "No such attribute '%s'");
  }

  public PythonVersion getDefaultPythonVersion() {
    return ruleContext.getRule().isAttrDefined("default_python_version", Type.STRING)
        ? getPythonVersionAttr(ruleContext)
        : null;
  }

  /** Returns the parsed value of the "srcs_version" attribute. */
  private static PythonVersion getSrcsVersionAttr(RuleContext ruleContext) {
    String attrValue = ruleContext.attributes().get("srcs_version", Type.STRING);
    try {
      return PythonVersion.parseSrcsValue(attrValue);
    } catch (IllegalArgumentException ex) {
      // Should already have been disallowed in the rule.
      ruleContext.attributeError(
          "srcs_version",
          String.format(
              "'%s' is not a valid value. Expected one of: %s",
              attrValue, Joiner.on(", ").join(PythonVersion.ALL_STRINGS)));
      return PythonVersion.DEFAULT_SRCS_VALUE;
    }
  }

  /** Returns the parsed value of the "default_python_version" attribute. */
  private static PythonVersion getPythonVersionAttr(RuleContext ruleContext) {
    String attrValue = ruleContext.attributes().get("default_python_version", Type.STRING);
    try {
      return PythonVersion.parseTargetValue(attrValue);
    } catch (IllegalArgumentException ex) {
      // Should already have been disallowed in the rule.
      ruleContext.attributeError(
          "default_python_version",
          String.format(
              "'%s' is not a valid value. Expected one of: %s",
              attrValue, Joiner.on(", ").join(PythonVersion.TARGET_STRINGS)));
      return PythonVersion.DEFAULT_TARGET_VALUE;
    }
  }

  /**
   * Returns a mutable List of the source Artifacts.
   */
  public List<Artifact> validateSrcs() {
    List<Artifact> sourceFiles = new ArrayList<>();
    // TODO(bazel-team): Need to get the transitive deps closure, not just the
    //                 sources of the rule.
    for (TransitiveInfoCollection src : ruleContext
        .getPrerequisitesIf("srcs", Mode.TARGET, FileProvider.class)) {
      // Make sure that none of the sources contain hyphens.
      if (Util.containsHyphen(src.getLabel().getPackageFragment())) {
        ruleContext.attributeError("srcs",
            src.getLabel() + ": paths to Python packages may not contain '-'");
      }
      Iterable<Artifact> pySrcs =
          FileType.filter(
              src.getProvider(FileProvider.class).getFilesToBuild(), PyRuleClasses.PYTHON_SOURCE);
      Iterables.addAll(sourceFiles, pySrcs);
      if (Iterables.isEmpty(pySrcs)) {
        ruleContext.attributeWarning("srcs",
            "rule '" + src.getLabel() + "' does not produce any Python source files");
      }
    }

    return convertedFiles != null
        ? ImmutableList.copyOf(convertedFiles.values())
        : sourceFiles;
  }

  /**
   * Checks that the package name of this Python rule does not contain a '-'.
   */
  void validatePackageName() {
    if (Util.containsHyphen(ruleContext.getLabel().getPackageFragment())) {
      ruleContext.ruleError("paths to Python packages may not contain '-'");
    }
  }

  /**
   * Adds a {@link PseudoAction} to the build graph that is only used
   * for providing information to the blaze extra_action feature.
   */
  void addPyExtraActionPseudoAction() {
    if (ruleContext.getConfiguration().getActionListeners().isEmpty()) {
      return;
    }

    // We need to do it in this convoluted way because we must not add the files declared in the
    // srcs of this rule. Note that it is not enough to remove the direct members from the nested
    // set of the current rule, because the same files may have been declared in a dependency, too.
    NestedSetBuilder<Artifact> depBuilder = NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFrom(getTargetDeps(), depBuilder);
    NestedSet<Artifact> dependencies = depBuilder.build();

    ruleContext.registerAction(
        makePyExtraActionPseudoAction(
            ruleContext.getActionOwner(),
            // Has to be unfiltered sources as filtered will give an error for
            // unsupported file types where as certain tests only expect a warning.
            ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list(),
            dependencies,
            PseudoAction.getDummyOutput(ruleContext)));
  }

  /**
   * Creates a {@link PseudoAction} that is only used for providing
   * information to the blaze extra_action feature.
   */
  public static Action makePyExtraActionPseudoAction(
      ActionOwner owner,
      Iterable<Artifact> sources,
      Iterable<Artifact> dependencies,
      Artifact output) {

    PythonInfo info =
        PythonInfo.newBuilder()
            .addAllSourceFile(Artifact.toExecPaths(sources))
            .addAllDepFile(Artifact.toExecPaths(dependencies))
            .build();

    return new PyPseudoAction(
        owner,
        NestedSetBuilder.wrap(Order.STABLE_ORDER, Iterables.concat(sources, dependencies)),
        ImmutableList.of(output),
        "Python",
        PYTHON_INFO,
        info);
  }

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final GeneratedExtension<ExtraActionInfo, PythonInfo> PYTHON_INFO = PythonInfo.pythonInfo;

  private void addSourceFiles(NestedSetBuilder<Artifact> builder, Iterable<Artifact> artifacts) {
    Preconditions.checkState(convertedFiles == null);
    if (sourcesVersion == PythonVersion.PY2 && version == PythonVersion.PY3) {
      convertedFiles = PythonUtils.generate2to3Actions(ruleContext, artifacts);
    }
    builder.addAll(artifacts);
  }

  private Iterable<? extends TransitiveInfoCollection> getTargetDeps() {
    return ruleContext.getPrerequisites("deps", Mode.TARGET);
  }

  private NestedSet<Artifact> getTransitivePythonSourcesFromSkylarkProvider(
      TransitiveInfoCollection dep) {
    StructImpl pythonSkylarkProvider = null;
    try {
      pythonSkylarkProvider =
          SkylarkType.cast(
              dep.get(PYTHON_SKYLARK_PROVIDER_NAME),
              StructImpl.class,
              null,
              "%s should be a struct",
              PYTHON_SKYLARK_PROVIDER_NAME);

      if (pythonSkylarkProvider != null) {
        Object sourceFiles = pythonSkylarkProvider.getValue(TRANSITIVE_PYTHON_SRCS);
        String errorType;
        if (sourceFiles == null) {
          errorType = "null";
        } else {
          errorType = EvalUtils.getDataTypeNameFromClass(sourceFiles.getClass());
        }
        String errorMsg = "Illegal Argument: attribute '%s' in provider '%s' is "
            + "of unexpected type. Should be a set, but got a '%s'";
        NestedSet<Artifact> pythonSourceFiles = SkylarkType.cast(
            sourceFiles, SkylarkNestedSet.class, Artifact.class, null,
            errorMsg, TRANSITIVE_PYTHON_SRCS, PYTHON_SKYLARK_PROVIDER_NAME, errorType)
            .getSet(Artifact.class);
        return pythonSourceFiles;
      }
    } catch (EvalException e) {
      ruleContext.ruleError(e.getMessage());
    }
    return null;
  }

  private void collectTransitivePythonSourcesFrom(
      Iterable<? extends TransitiveInfoCollection> deps, NestedSetBuilder<Artifact> builder) {
    for (TransitiveInfoCollection dep : deps) {
      NestedSet<Artifact> pythonSourceFiles = getTransitivePythonSourcesFromSkylarkProvider(dep);
      if (pythonSourceFiles != null) {
        builder.addTransitive(pythonSourceFiles);
      } else {
        // TODO(bazel-team): We also collect .py source files from deps (e.g. for proto_library
        // rules). Rules should implement PythonSourcesProvider instead.
        FileProvider provider = dep.getProvider(FileProvider.class);
        builder.addAll(FileType.filter(provider.getFilesToBuild(), PyRuleClasses.PYTHON_SOURCE));
      }
    }
  }

  private NestedSet<Artifact> collectTransitivePythonSources() {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFrom(getTargetDeps(), builder);
    addSourceFiles(builder,
        ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET)
            .filter(PyRuleClasses.PYTHON_SOURCE).list());
    return builder.build();
  }

  public NestedSet<Artifact> collectTransitivePythonSourcesWithoutLocal() {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFrom(getTargetDeps(), builder);
    return builder.build();
  }

  public NestedSet<String> collectImports(RuleContext ruleContext, PythonSemantics semantics) {
    NestedSetBuilder<String> builder = NestedSetBuilder.compileOrder();
    builder.addAll(semantics.getImports(ruleContext));
    collectTransitivePythonImports(builder);
    return builder.build();
  }

  private void collectTransitivePythonImports(NestedSetBuilder<String> builder) {
    for (TransitiveInfoCollection dep : getTargetDeps()) {
      if (dep.getProvider(PythonImportsProvider.class) != null) {
        PythonImportsProvider provider = dep.getProvider(PythonImportsProvider.class);
        builder.addTransitive(provider.getTransitivePythonImports());
      }
    }
  }

  /**
   * Checks that the source file version is compatible with the Python interpreter.
   */
  private void checkSourceIsCompatible(PythonVersion targetVersion, PythonVersion sourceVersion,
                                          Label source) {
    // Treat PY3 as PY3ONLY: we'll never implement 3to2.
    if ((targetVersion == PythonVersion.PY2 || targetVersion == PythonVersion.PY2AND3)
        && (sourceVersion == PythonVersion.PY3 || sourceVersion == PythonVersion.PY3ONLY)) {
      ruleContext.ruleError("Rule '" + source
          + "' can only be used with Python 3, and cannot be converted to Python 2");
    }
    if ((targetVersion == PythonVersion.PY3 || targetVersion == PythonVersion.PY2AND3)
        && sourceVersion == PythonVersion.PY2ONLY) {
      ruleContext.ruleError(
          "Rule '"
              + source
              + "' can only be used with Python 2, and cannot be converted to Python 3");
    }
  }

  /** @return A String that is the full path to the main python entry point. */
  public String determineMainExecutableSource(boolean withWorkspaceName) {
    String mainSourceName;
    Rule target = ruleContext.getRule();
    boolean explicitMain = target.isAttributeValueExplicitlySpecified("main");
    if (explicitMain) {
      mainSourceName = ruleContext.attributes().get("main", BuildType.LABEL).getName();
      if (!mainSourceName.endsWith(".py")) {
        ruleContext.attributeError("main", "main must end in '.py'");
      }
    } else {
      String ruleName = target.getName();
      if (ruleName.endsWith(".py")) {
        ruleContext.attributeError("name", "name must not end in '.py'");
      }
      mainSourceName = ruleName + ".py";
    }
    PathFragment mainSourcePath = PathFragment.create(mainSourceName);

    Artifact mainArtifact = null;
    for (Artifact outItem : ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list()) {
      if (outItem.getRootRelativePath().endsWith(mainSourcePath)) {
        if (mainArtifact == null) {
          mainArtifact = outItem;
        } else {
          ruleContext.attributeError("srcs",
              buildMultipleMainMatchesErrorText(explicitMain, mainSourceName,
                  mainArtifact.getRunfilesPath().toString(),
                  outItem.getRunfilesPath().toString()));
        }
      }
    }

    if (mainArtifact == null) {
      ruleContext.attributeError("srcs", buildNoMainMatchesErrorText(explicitMain, mainSourceName));
      return null;
    }
    if (!withWorkspaceName) {
      return mainArtifact.getRunfilesPath().getPathString();
    }
    PathFragment workspaceName =
        PathFragment.create(ruleContext.getRule().getPackage().getWorkspaceName());
    return workspaceName.getRelative(mainArtifact.getRunfilesPath()).getPathString();
  }

  public String determineMainExecutableSource() {
    return determineMainExecutableSource(true);
  }

  public Artifact getExecutable() {
    return executable;
  }

  public Map<PathFragment, Artifact> getConvertedFiles() {
    return convertedFiles;
  }

  public NestedSet<Artifact> getFilesToBuild() {
    return filesToBuild;
  }

  public boolean usesSharedLibraries() {
    try {
      return checkForSharedLibraries(Iterables.concat(
              ruleContext.getPrerequisites("deps", Mode.TARGET),
              ruleContext.getPrerequisites("data", Mode.DONT_CHECK)));
    } catch (EvalException e) {
      ruleContext.ruleError(e.getMessage());
      return false;
    }
  }


  /**
   * Returns true if this target has an .so file in its transitive dependency closure.
   */
  public static boolean checkForSharedLibraries(Iterable<TransitiveInfoCollection> deps)
          throws EvalException{
    for (TransitiveInfoCollection dep : deps) {
      Object providerObject = dep.get(PYTHON_SKYLARK_PROVIDER_NAME);
      if (providerObject != null) {
        SkylarkType.checkType(providerObject, StructImpl.class, null);
        StructImpl provider = (StructImpl) providerObject;
        Boolean isUsingSharedLibrary = provider.getValue(IS_USING_SHARED_LIBRARY, Boolean.class);
        if (Boolean.TRUE.equals(isUsingSharedLibrary)) {
          return true;
        }
      } else if (FileType.contains(
          dep.getProvider(FileProvider.class).getFilesToBuild(), CppFileTypes.SHARED_LIBRARY)) {
        return true;
      }
    }

    return false;
  }

  private static String buildMultipleMainMatchesErrorText(boolean explicit, String proposedMainName,
      String match1, String match2) {
    String errorText;
    if (explicit) {
      errorText = "file name '" + proposedMainName
          + "' specified by 'main' attribute matches multiple files: e.g., '" + match1
          + "' and '" + match2 + "'";
    } else {
      errorText = "default main file name '" + proposedMainName
          + "' matches multiple files.  Perhaps specify an explicit file with 'main' attribute?  "
          + "Matches were: '" + match1 + "' and '" + match2 + "'";
    }
    return errorText;
  }

  private static String buildNoMainMatchesErrorText(boolean explicit, String proposedMainName) {
    String errorText;
    if (explicit) {
      errorText = "could not find '" + proposedMainName
          + "' as specified by 'main' attribute";
    } else {
      errorText = "corresponding default '" + proposedMainName + "' does not appear in srcs. Add it"
          + " or override default file name with a 'main' attribute";
    }
    return errorText;
  }

  // Used purely to set the legacy ActionType of the ExtraActionInfo.
  @Immutable
  private static final class PyPseudoAction extends PseudoAction<PythonInfo> {
    private static final UUID ACTION_UUID = UUID.fromString("8d720129-bc1a-481f-8c4c-dbe11dcef319");

    public PyPseudoAction(ActionOwner owner,
        NestedSet<Artifact> inputs, Collection<Artifact> outputs,
        String mnemonic, GeneratedExtension<ExtraActionInfo, PythonInfo> infoExtension,
        PythonInfo info) {
      super(ACTION_UUID, owner, inputs, outputs, mnemonic, infoExtension, info);
    }
  }
}
