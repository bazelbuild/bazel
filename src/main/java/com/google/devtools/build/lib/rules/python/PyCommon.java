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
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.EvalException;
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

/** A helper class for analyzing a Python configured target. */
public final class PyCommon {

  public static final String DEFAULT_PYTHON_VERSION_ATTRIBUTE = "default_python_version";
  public static final String PYTHON_VERSION_ATTRIBUTE = "python_version";

  private static final LocalMetadataCollector METADATA_COLLECTOR = new LocalMetadataCollector() {
    @Override
    public void collectMetadataArtifacts(Iterable<Artifact> artifacts,
        AnalysisEnvironment analysisEnvironment, NestedSetBuilder<Artifact> metadataFilesBuilder) {
      // Python doesn't do any compilation, so we simply return the empty set.
    }
  };

  private final RuleContext ruleContext;

  /**
   * The Python major version for which this target is being built, as per the {@code
   * python_version} attribute or the configuration.
   *
   * <p>This is always either {@code PY2} or {@code PY3}.
   */
  private final PythonVersion version;

  /**
   * The level of compatibility with Python major versions, as per the {@code srcs_version}
   * attribute.
   */
  private final PythonVersion sourcesVersion;

  /**
   * The Python sources belonging to this target's transitive {@code deps}, not including this
   * target's own {@code srcs}.
   */
  private final NestedSet<Artifact> dependencyTransitivePythonSources;

  /**
   * The Python sources belonging to this target's transitive {@code deps}, including the Python
   * sources in this target's {@code srcs}.
   */
  private final NestedSet<Artifact> transitivePythonSources;

  /** Whether this target or any of its {@code deps} or {@code data} deps has a shared library. */
  private final boolean usesSharedLibraries;

  /**
   * Symlink map from root-relative paths to 2to3 converted source artifacts.
   *
   * <p>Null if no 2to3 conversion is required.
   */
  private final Map<PathFragment, Artifact> convertedFiles;

  private Artifact executable = null;

  private NestedSet<Artifact> filesToBuild = null;

  public PyCommon(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.sourcesVersion = getSrcsVersionAttr(ruleContext);
    this.version = ruleContext.getFragment(PythonConfiguration.class).getPythonVersion();
    this.dependencyTransitivePythonSources = collectDependencyTransitivePythonSources();
    this.transitivePythonSources = collectTransitivePythonSources();
    this.usesSharedLibraries = checkForSharedLibraries();
    this.convertedFiles = maybeMakeConvertedFiles();
    checkSourceIsCompatible(this.version, this.sourcesVersion, ruleContext.getLabel());
    validatePythonVersionAttr(ruleContext);
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
            PyProvider.PROVIDER_NAME,
            PyProvider.builder()
                .setTransitiveSources(transitivePythonSources)
                .setUsesSharedLibraries(usesSharedLibraries)
                .setImports(imports)
                .build())
        // Python targets are not really compilable. The best we can do is make sure that all
        // generated source files are ready.
        .addOutputGroup(OutputGroupInfo.FILES_TO_COMPILE, transitivePythonSources)
        .addOutputGroup(OutputGroupInfo.COMPILATION_PREREQUISITES, transitivePythonSources);
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

  /**
   * Reports an attribute error if the {@code default_python_version} attribute is set but
   * disallowed by the configuration.
   */
  private static void validatePythonVersionAttr(RuleContext ruleContext) {
    AttributeMap attrs = ruleContext.attributes();
    PythonConfiguration config = ruleContext.getFragment(PythonConfiguration.class);
    if (attrs.has(DEFAULT_PYTHON_VERSION_ATTRIBUTE, Type.STRING)
        && attrs.isAttributeValueExplicitlySpecified(DEFAULT_PYTHON_VERSION_ATTRIBUTE)
        && !config.oldPyVersionApiAllowed()) {
      ruleContext.attributeError(
          DEFAULT_PYTHON_VERSION_ATTRIBUTE,
          "the 'default_python_version' attribute is disabled by the "
              + "'--experimental_remove_old_python_version_api' flag");
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
   * Adds a {@link PseudoAction} to the build graph that is only used for providing information to
   * the blaze extra_action feature.
   */
  void addPyExtraActionPseudoAction() {
    if (ruleContext.getConfiguration().getActionListeners().isEmpty()) {
      return;
    }
    ruleContext.registerAction(
        makePyExtraActionPseudoAction(
            ruleContext.getActionOwner(),
            // Has to be unfiltered sources as filtered will give an error for
            // unsupported file types where as certain tests only expect a warning.
            ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list(),
            // We must not add the files declared in the srcs of this rule.;
            dependencyTransitivePythonSources,
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

  /**
   * If 2to3 conversion is to be done, creates the 2to3 actions and returns the map of converted
   * files; otherwise returns null.
   */
  // TODO(#1393): 2to3 conversion doesn't work in Bazel and the attempt to invoke it for Bazel
  // should be removed / factored away into PythonSemantics.
  private Map<PathFragment, Artifact> maybeMakeConvertedFiles() {
    if (sourcesVersion == PythonVersion.PY2 && version == PythonVersion.PY3) {
      Iterable<Artifact> artifacts =
          ruleContext
              .getPrerequisiteArtifacts("srcs", Mode.TARGET)
              .filter(PyRuleClasses.PYTHON_SOURCE)
              .list();
      return PythonUtils.generate2to3Actions(ruleContext, artifacts);
    } else {
      return null;
    }
  }

  private static String getOrderErrorMessage(String fieldName, Order expected, Order actual) {
    return String.format(
        "Incompatible order for %s: expected 'default' or '%s', got '%s'",
        fieldName, expected.getSkylarkName(), actual.getSkylarkName());
  }

  private void collectTransitivePythonSourcesFrom(
      Iterable<? extends TransitiveInfoCollection> deps, NestedSetBuilder<Artifact> builder) {
    for (TransitiveInfoCollection dep : deps) {
      if (PyProvider.hasProvider(dep)) {
        try {
          StructImpl info = PyProvider.getProvider(dep);
          NestedSet<Artifact> sources = PyProvider.getTransitiveSources(info);
          if (!builder.getOrder().isCompatible(sources.getOrder())) {
            ruleContext.ruleError(
                getOrderErrorMessage(
                    PyProvider.TRANSITIVE_SOURCES, builder.getOrder(), sources.getOrder()));
          } else {
            builder.addTransitive(sources);
          }
        } catch (EvalException e) {
          // Either the provider type or field type is bad.
          ruleContext.ruleError(e.getMessage());
        }
      } else {
        // TODO(bazel-team): We also collect .py source files from deps (e.g. for proto_library
        // rules). We should have rules implement a "PythonSourcesProvider" instead.
        FileProvider provider = dep.getProvider(FileProvider.class);
        builder.addAll(FileType.filter(provider.getFilesToBuild(), PyRuleClasses.PYTHON_SOURCE));
      }
    }
  }

  private NestedSet<Artifact> collectTransitivePythonSources() {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFrom(ruleContext.getPrerequisites("deps", Mode.TARGET), builder);
    builder.addAll(
        ruleContext
            .getPrerequisiteArtifacts("srcs", Mode.TARGET)
            .filter(PyRuleClasses.PYTHON_SOURCE)
            .list());
    return builder.build();
  }

  private NestedSet<Artifact> collectDependencyTransitivePythonSources() {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFrom(ruleContext.getPrerequisites("deps", Mode.TARGET), builder);
    return builder.build();
  }

  /** Returns the transitive Python sources collected from the deps and srcs attributes. */
  public NestedSet<Artifact> getTransitivePythonSources() {
    return transitivePythonSources;
  }

  /**
   * Returns the transitive Python sources collected from the deps attribute, not including sources
   * from the srcs attribute (unless they were separately reached via deps).
   */
  public NestedSet<Artifact> getDependencyTransitivePythonSources() {
    return dependencyTransitivePythonSources;
  }

  public NestedSet<String> collectImports(RuleContext ruleContext, PythonSemantics semantics) {
    NestedSetBuilder<String> builder = NestedSetBuilder.compileOrder();
    builder.addAll(semantics.getImports(ruleContext));
    collectTransitivePythonImports(builder);
    return builder.build();
  }

  private void collectTransitivePythonImports(NestedSetBuilder<String> builder) {
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
      if (dep.getProvider(PythonImportsProvider.class) != null) {
        PythonImportsProvider provider = dep.getProvider(PythonImportsProvider.class);
        NestedSet<String> imports = provider.getTransitivePythonImports();
        if (!builder.getOrder().isCompatible(imports.getOrder())) {
          // TODO(brandjon): Add test case for this error, once we replace PythonImportsProvider
          // with the normal Python provider and once we clean up our provider merge logic.
          ruleContext.ruleError(
              getOrderErrorMessage(PyProvider.IMPORTS, builder.getOrder(), imports.getOrder()));
        } else {
          builder.addTransitive(imports);
        }
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

  /**
   * Returns true if any of this target's {@code deps} or {@code data} deps has a shared library
   * file (e.g. a {@code .so}) in its transitive dependency closure.
   *
   * <p>For targets with the py provider, we consult the {@code uses_shared_libraries} field. For
   * targets without this provider, we look for {@link CppFileTypes#SHARED_LIBRARY}-type files in
   * the filesToBuild.
   */
  private boolean checkForSharedLibraries() {
    Iterable<? extends TransitiveInfoCollection> targets;
    // For all rule types that use PyCommon, deps is a required attribute but not data.
    if (ruleContext.attributes().has("data")) {
      targets =
          Iterables.concat(
              ruleContext.getPrerequisites("deps", Mode.TARGET),
              ruleContext.getPrerequisites("data", Mode.DONT_CHECK));
    } else {
      targets = ruleContext.getPrerequisites("deps", Mode.TARGET);
    }
    try {
      for (TransitiveInfoCollection target : targets) {
        if (checkForSharedLibraries(target)) {
          return true;
        }
      }
      return false;
    } catch (EvalException e) {
      ruleContext.ruleError(e.getMessage());
      return false;
    }
  }

  private static boolean checkForSharedLibraries(TransitiveInfoCollection target)
      throws EvalException {
    if (PyProvider.hasProvider(target)) {
      return PyProvider.getUsesSharedLibraries(PyProvider.getProvider(target));
    } else {
      NestedSet<Artifact> files = target.getProvider(FileProvider.class).getFilesToBuild();
      return FileType.contains(files, CppFileTypes.SHARED_LIBRARY);
    }
  }

  public boolean usesSharedLibraries() {
    return usesSharedLibraries;
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
