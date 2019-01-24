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
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.PythonInfo;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.Util;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.LocalMetadataCollector;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
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

  /**
   * Returns the Python version based on the {@code python_version} and {@code
   * default_python_version} attributes of the given {@code AttributeMap}.
   *
   * <p>It is expected that both attributes are defined, string-typed, and default to {@link
   * PythonVersion#_INTERNAL_SENTINEL}. The returned version is the value of {@code python_version}
   * if it is not the sentinel, then {@code default_python_version} if it is not the sentinel, then
   * finally {@code default}. In all cases the return value is a target version value (either {@code
   * PY2} or {@code PY3}).
   *
   * @throws IllegalArgumentException if the attributes are not present, not string-typed, or not
   *     parsable as target {@link PythonVersion} values or as the sentinel value; or if {@code
   *     default} is not a target version value
   */
  public static PythonVersion readPythonVersionFromAttributes(
      AttributeMap attrs, PythonVersion defaultVersion) {
    Preconditions.checkArgument(defaultVersion.isTargetValue());
    PythonVersion pythonVersionAttr =
        PythonVersion.parseTargetOrSentinelValue(attrs.get(PYTHON_VERSION_ATTRIBUTE, Type.STRING));
    PythonVersion defaultPythonVersionAttr =
        PythonVersion.parseTargetOrSentinelValue(
            attrs.get(DEFAULT_PYTHON_VERSION_ATTRIBUTE, Type.STRING));
    if (pythonVersionAttr != PythonVersion._INTERNAL_SENTINEL) {
      return pythonVersionAttr;
    } else if (defaultPythonVersionAttr != PythonVersion._INTERNAL_SENTINEL) {
      return defaultPythonVersionAttr;
    } else {
      return defaultVersion;
    }
  }

  private static final LocalMetadataCollector METADATA_COLLECTOR = new LocalMetadataCollector() {
    @Override
    public void collectMetadataArtifacts(Iterable<Artifact> artifacts,
        AnalysisEnvironment analysisEnvironment, NestedSetBuilder<Artifact> metadataFilesBuilder) {
      // Python doesn't do any compilation, so we simply return the empty set.
    }
  };

  /** The context for the target this {@code PyCommon} is helping to analyze. */
  private final RuleContext ruleContext;

  /** The pluggable semantics object with hooks that customizes how analysis is done. */
  private final PythonSemantics semantics;

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

  /** Extra Python module import paths propagated or used by this target. */
  private final NestedSet<String> imports;

  /**
   * Whether any of this target's transitive {@code deps} have PY2-only source files, including this
   * target itself.
   */
  private final boolean hasPy2OnlySources;

  /**
   * Whether any of this target's transitive {@code deps} have PY3-only source files, including this
   * target itself.
   */
  private final boolean hasPy3OnlySources;

  /**
   * Symlink map from root-relative paths to 2to3 converted source artifacts.
   *
   * <p>Null if no 2to3 conversion is required.
   */
  private final Map<PathFragment, Artifact> convertedFiles;

  private Artifact executable = null;

  private NestedSet<Artifact> filesToBuild = null;

  private static String getOrderErrorMessage(String fieldName, Order expected, Order actual) {
    return String.format(
        "Incompatible order for %s: expected 'default' or '%s', got '%s'",
        fieldName, expected.getSkylarkName(), actual.getSkylarkName());
  }

  public PyCommon(RuleContext ruleContext, PythonSemantics semantics) {
    this.ruleContext = ruleContext;
    this.semantics = semantics;
    this.sourcesVersion = initSrcsVersionAttr(ruleContext);
    this.version = ruleContext.getFragment(PythonConfiguration.class).getPythonVersion();
    this.dependencyTransitivePythonSources = initDependencyTransitivePythonSources(ruleContext);
    this.transitivePythonSources = initTransitivePythonSources(ruleContext);
    this.usesSharedLibraries = initUsesSharedLibraries(ruleContext);
    this.imports = initImports(ruleContext, semantics);
    this.hasPy2OnlySources = initHasPy2OnlySources(ruleContext, this.sourcesVersion);
    this.hasPy3OnlySources = initHasPy3OnlySources(ruleContext, this.sourcesVersion);
    this.convertedFiles = makeAndInitConvertedFiles(ruleContext, version, this.sourcesVersion);
    maybeValidateVersionCompatibleWithOwnSourcesAttr();
    validateTargetPythonVersionAttr(DEFAULT_PYTHON_VERSION_ATTRIBUTE);
    validateTargetPythonVersionAttr(PYTHON_VERSION_ATTRIBUTE);
    validateOldVersionAttrNotUsedIfDisabled();
  }

  /** Returns the parsed value of the "srcs_version" attribute. */
  private static PythonVersion initSrcsVersionAttr(RuleContext ruleContext) {
    String attrValue = ruleContext.attributes().get("srcs_version", Type.STRING);
    try {
      return PythonVersion.parseSrcsValue(attrValue);
    } catch (IllegalArgumentException ex) {
      // Should already have been disallowed in the rule.
      ruleContext.attributeError(
          "srcs_version",
          String.format(
              "'%s' is not a valid value. Expected one of: %s",
              attrValue, Joiner.on(", ").join(PythonVersion.SRCS_STRINGS)));
      return PythonVersion.DEFAULT_SRCS_VALUE;
    }
  }

  private static NestedSet<Artifact> initDependencyTransitivePythonSources(
      RuleContext ruleContext) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFromDeps(ruleContext, builder);
    return builder.build();
  }

  private static NestedSet<Artifact> initTransitivePythonSources(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.compileOrder();
    collectTransitivePythonSourcesFromDeps(ruleContext, builder);
    builder.addAll(
        ruleContext
            .getPrerequisiteArtifacts("srcs", Mode.TARGET)
            .filter(PyRuleClasses.PYTHON_SOURCE)
            .list());
    return builder.build();
  }

  /**
   * Gathers transitive .py files from {@code deps} (not including this target's {@code srcs} and
   * adds them to {@code builder}.
   */
  private static void collectTransitivePythonSourcesFromDeps(
      RuleContext ruleContext, NestedSetBuilder<Artifact> builder) {
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
      if (PyStructUtils.hasProvider(dep)) {
        try {
          StructImpl info = PyStructUtils.getProvider(dep);
          NestedSet<Artifact> sources = PyStructUtils.getTransitiveSources(info);
          builder.addTransitive(sources);
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

  /**
   * Returns true if any of this target's {@code deps} or {@code data} deps has a shared library
   * file (e.g. a {@code .so}) in its transitive dependency closure.
   *
   * <p>For targets with the py provider, we consult the {@code uses_shared_libraries} field. For
   * targets without this provider, we look for {@link CppFileTypes#SHARED_LIBRARY}-type files in
   * the filesToBuild.
   */
  private static boolean initUsesSharedLibraries(RuleContext ruleContext) {
    Iterable<? extends TransitiveInfoCollection> targets;
    // The deps attribute must exist for all rule types that use PyCommon, but not necessarily the
    // data attribute.
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
    if (PyStructUtils.hasProvider(target)) {
      return PyStructUtils.getUsesSharedLibraries(PyStructUtils.getProvider(target));
    } else {
      NestedSet<Artifact> files = target.getProvider(FileProvider.class).getFilesToBuild();
      return FileType.contains(files, CppFileTypes.SHARED_LIBRARY);
    }
  }

  private static NestedSet<String> initImports(RuleContext ruleContext, PythonSemantics semantics) {
    NestedSetBuilder<String> builder = NestedSetBuilder.compileOrder();
    builder.addAll(semantics.getImports(ruleContext));

    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
      if (dep.getProvider(PythonImportsProvider.class) != null) {
        PythonImportsProvider provider = dep.getProvider(PythonImportsProvider.class);
        NestedSet<String> imports = provider.getTransitivePythonImports();
        if (!builder.getOrder().isCompatible(imports.getOrder())) {
          // TODO(brandjon): Add test case for this error, once we replace PythonImportsProvider
          // with the normal Python provider and once we clean up our provider merge logic.
          ruleContext.ruleError(
              getOrderErrorMessage(PyStructUtils.IMPORTS, builder.getOrder(), imports.getOrder()));
        } else {
          builder.addTransitive(imports);
        }
      }
    }

    return builder.build();
  }

  /**
   * Returns true if any of {@code deps} has a py provider with {@code has_py2_only_sources} set, or
   * this target has a {@code srcs_version} of {@code PY2ONLY}.
   */
  // TODO(#1393): For Bazel, deprecate 2to3 support and treat PY2 the same as PY2ONLY.
  private static boolean initHasPy2OnlySources(
      RuleContext ruleContext, PythonVersion sourcesVersion) {
    if (sourcesVersion == PythonVersion.PY2ONLY) {
      return true;
    }
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
      if (PyStructUtils.hasProvider(dep)) {
        try {
          if (PyStructUtils.getHasPy2OnlySources(PyStructUtils.getProvider(dep))) {
            return true;
          }
        } catch (EvalException e) {
          ruleContext.ruleError(String.format("In dep '%s': %s", dep.getLabel(), e.getMessage()));
        }
      }
    }
    return false;
  }

  /**
   * Returns true if any of {@code deps} has a py provider with {@code has_py3_only_sources} set, or
   * this target has {@code srcs_version} of {@code PY3} or {@code PY3ONLY}.
   */
  private static boolean initHasPy3OnlySources(
      RuleContext ruleContext, PythonVersion sourcesVersion) {
    if (sourcesVersion == PythonVersion.PY3 || sourcesVersion == PythonVersion.PY3ONLY) {
      return true;
    }
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
      if (PyStructUtils.hasProvider(dep)) {
        try {
          if (PyStructUtils.getHasPy3OnlySources(PyStructUtils.getProvider(dep))) {
            return true;
          }
        } catch (EvalException e) {
          ruleContext.ruleError(String.format("In dep '%s': %s", dep.getLabel(), e.getMessage()));
        }
      }
    }
    return false;
  }

  /**
   * If 2to3 conversion is to be done, creates the 2to3 actions and returns the map of converted
   * files; otherwise returns null.
   */
  // TODO(#1393): 2to3 conversion doesn't work in Bazel and the attempt to invoke it for Bazel
  // should be removed / factored away into PythonSemantics.
  private static Map<PathFragment, Artifact> makeAndInitConvertedFiles(
      RuleContext ruleContext, PythonVersion version, PythonVersion sourcesVersion) {
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

  /**
   * Under the old version semantics ({@code
   * --experimental_allow_python_version_transitions=false}), checks that the {@code srcs_version}
   * attribute is compatible with the Python version as determined by the configuration.
   *
   * <p>A failure is reported as a rule error.
   *
   * <p>This check is local to the current target and intended to be enforced by each {@code
   * py_library} up the dependency chain.
   *
   * <p>No-op under the new version semantics.
   */
  private void maybeValidateVersionCompatibleWithOwnSourcesAttr() {
    if (ruleContext.getFragment(PythonConfiguration.class).useNewPyVersionSemantics()) {
      return;
    }
    // Treat PY3 as PY3ONLY: we'll never implement 3to2.
    if ((version == PythonVersion.PY2 || version == PythonVersion.PY2AND3)
        && (sourcesVersion == PythonVersion.PY3 || sourcesVersion == PythonVersion.PY3ONLY)) {
      ruleContext.ruleError(
          "Rule '"
              + ruleContext.getLabel()
              + "' can only be used with Python 3, and cannot be converted to Python 2");
    }
    if ((version == PythonVersion.PY3 || version == PythonVersion.PY2AND3)
        && sourcesVersion == PythonVersion.PY2ONLY) {
      ruleContext.ruleError(
          "Rule '"
              + ruleContext.getLabel()
              + "' can only be used with Python 2, and cannot be converted to Python 3");
    }
  }

  /**
   * Reports an attribute error if the given target Python version attribute ({@code
   * default_python_version} or {@code python_version}) cannot be parsed as {@code PY2}, {@code
   * PY3}, or the sentinel value.
   *
   * <p>This *should* be enforced by rule attribute validation ({@link
   * Attribute.Builder.allowedValues}), but this check is here to fail-fast just in case.
   */
  private void validateTargetPythonVersionAttr(String attr) {
    AttributeMap attrs = ruleContext.attributes();
    if (!attrs.has(attr, Type.STRING)) {
      return;
    }
    String attrValue = attrs.get(attr, Type.STRING);
    try {
      PythonVersion.parseTargetOrSentinelValue(attrValue);
    } catch (IllegalArgumentException ex) {
      ruleContext.attributeError(
          attr,
          String.format("'%s' is not a valid value. Expected either 'PY2' or 'PY3'", attrValue));
    }
  }

  /**
   * Reports an attribute error if the {@code default_python_version} attribute is set but
   * disallowed by the configuration.
   */
  private void validateOldVersionAttrNotUsedIfDisabled() {
    AttributeMap attrs = ruleContext.attributes();
    if (!attrs.has(DEFAULT_PYTHON_VERSION_ATTRIBUTE, Type.STRING)) {
      return;
    }
    PythonVersion value;
    try {
      value =
          PythonVersion.parseTargetOrSentinelValue(
              attrs.get(DEFAULT_PYTHON_VERSION_ATTRIBUTE, Type.STRING));
    } catch (IllegalArgumentException e) {
      // Should be reported by validateTargetPythonVersionAttr(); no action required here.
      return;
    }
    PythonConfiguration config = ruleContext.getFragment(PythonConfiguration.class);
    if (value != PythonVersion._INTERNAL_SENTINEL && !config.oldPyVersionApiAllowed()) {
      ruleContext.attributeError(
          DEFAULT_PYTHON_VERSION_ATTRIBUTE,
          "the 'default_python_version' attribute is disabled by the "
              + "'--experimental_remove_old_python_version_api' flag");
    }
  }

  /**
   * Under the new version semantics ({@code --experimental_allow_python_version_transitions=true}),
   * if the Python version (as determined by the configuration) is inconsistent with {@link
   * #hasPy2OnlySources} or {@link #hasPy3OnlySources}, emits a {@link FailAction} that "generates"
   * the executable.
   *
   * <p>If the version is consistent, or if we are using the old semantics, no such action is
   * emitted.
   *
   * <p>We use a {@code FailAction} rather than a rule error because we want to defer the error
   * until the execution phase. This way, we still get a configured target that the user can query
   * over with an aspect to find the exact transitive dependency that introduced the offending
   * version constraint.
   *
   * @return true if a {@link FailAction} was created
   */
  private boolean maybeCreateFailActionDueToTransitiveSourcesVersion() {
    if (!ruleContext.getFragment(PythonConfiguration.class).useNewPyVersionSemantics()) {
      return false;
    }
    // TODO(brandjon): Add hints to the error message about how to locate the offending
    // dependencies.
    String error = null;
    if (version == PythonVersion.PY2 && hasPy3OnlySources) {
      error =
          "target is being built for Python 2 but (transitively) includes Python 3-only sources";
    } else if (version == PythonVersion.PY3 && hasPy2OnlySources) {
      error =
          "target is being built for Python 3 but (transitively) includes Python 2-only sources";
    }
    if (error == null) {
      return false;
    } else {
      ruleContext.registerAction(
          new FailAction(ruleContext.getActionOwner(), ImmutableList.of(executable), error));
      return true;
    }
  }

  public PythonVersion getVersion() {
    return version;
  }

  public PythonVersion getSourcesVersion() {
    return sourcesVersion;
  }

  /**
   * Returns the transitive Python sources collected from the deps attribute, not including sources
   * from the srcs attribute (unless they were separately reached via deps).
   */
  public NestedSet<Artifact> getDependencyTransitivePythonSources() {
    return dependencyTransitivePythonSources;
  }

  /** Returns the transitive Python sources collected from the deps and srcs attributes. */
  public NestedSet<Artifact> getTransitivePythonSources() {
    return transitivePythonSources;
  }

  public boolean usesSharedLibraries() {
    return usesSharedLibraries;
  }

  public NestedSet<String> getImports() {
    return imports;
  }

  public boolean hasPy2OnlySources() {
    return hasPy2OnlySources;
  }

  public boolean hasPy3OnlySources() {
    return hasPy3OnlySources;
  }

  public Map<PathFragment, Artifact> getConvertedFiles() {
    return convertedFiles;
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
            PyStructUtils.PROVIDER_NAME,
            PyStructUtils.builder()
                .setTransitiveSources(transitivePythonSources)
                .setUsesSharedLibraries(usesSharedLibraries)
                .setImports(imports)
                .setHasPy2OnlySources(hasPy2OnlySources)
                .setHasPy3OnlySources(hasPy3OnlySources)
                .build())
        // Python targets are not really compilable. The best we can do is make sure that all
        // generated source files are ready.
        .addOutputGroup(OutputGroupInfo.FILES_TO_COMPILE, transitivePythonSources)
        .addOutputGroup(OutputGroupInfo.COMPILATION_PREREQUISITES, transitivePythonSources);
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

  public NestedSet<Artifact> getFilesToBuild() {
    return filesToBuild;
  }

  /**
   * Creates the actual executable artifact, i.e., emits a generating action for {@link
   * #getExecutable()}.
   *
   * <p>If there is a transitive sources version conflict, may produce a {@link FailAction} to
   * trigger an execution-time failure. See {@link
   * #maybeCreateFailActionDueToTransitiveSourcesVersion}.
   */
  public Artifact createExecutable(
      CcInfo ccInfo, NestedSet<String> givenImports, Runfiles.Builder defaultRunfilesBuilder)
      throws InterruptedException, RuleErrorException {
    boolean failed = maybeCreateFailActionDueToTransitiveSourcesVersion();
    if (failed) {
      return executable;
    } else {
      // TODO(#7054): We pass imports as an arg instead of taking them from the PyCommon field
      // because the imports logic is a little inconsistent, and passing it explicitly may help
      // avoid creating bugs that make the situation worse. We can eliminate this arg when we
      // straighten up the other imports logic.
      return semantics.createExecutable(
          ruleContext, this, ccInfo, givenImports, defaultRunfilesBuilder);
    }
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
