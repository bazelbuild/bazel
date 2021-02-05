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

import static net.starlark.java.eval.Starlark.NONE;

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
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.Util;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.server.FailureDetails.FailAction.Code;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.GeneratedMessage.GeneratedExtension;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** A helper class for analyzing a Python configured target. */
public final class PyCommon {

  /** Name of the version attribute. */
  public static final String PYTHON_VERSION_ATTRIBUTE = "python_version";

  /**
   * Returns the Python version based on the {@code python_version} attribute of the given {@code
   * AttributeMap}.
   *
   * <p>It is expected that the attribute is defined, string-typed, and defaults to {@link
   * PythonVersion#_INTERNAL_SENTINEL}. The returned version is the value of {@code python_version}
   * if it is not the sentinel (in which case it is either {@code PY2} or {@code PY3}), or null if
   * it is the sentinel.
   *
   * @throws IllegalArgumentException if the attribute is not present, not string-typed, or not
   *     parsable as a target {@link PythonVersion} value or the sentinel value
   */
  @Nullable
  public static PythonVersion readPythonVersionFromAttribute(AttributeMap attrs) {
    PythonVersion pythonVersionAttr =
        PythonVersion.parseTargetOrSentinelValue(attrs.get(PYTHON_VERSION_ATTRIBUTE, Type.STRING));
    return pythonVersionAttr != PythonVersion._INTERNAL_SENTINEL ? pythonVersionAttr : null;
  }

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

  /**
   * The Python sources from this target's {@code srcs}.
   *
   * <p>This is computed slightly differently than the difference between {@link
   * #dependencyTransitivePythonSources} and {@link transitivePythonSources}; it includes the
   * filesToBuild rather than the prerequisite artifacts.
   */
  // TODO(bazel-team): Can this be simplified to instead just be (transitivePythonSources -
  // dependencyTransitivePythonSources)?
  private final List<Artifact> directPythonSources;

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
   * Information about the runtime, as obtained from the toolchain.
   *
   * <p>This is non-null only if
   *
   * <ol>
   *   <li>the configuration says to pull the runtime from the toolchain (rather than from the
   *       legacy flags),
   *   <li>the target defines the attribute "$py_toolchain_type" (in which case it MUST also declare
   *       that it requires the Python toolchain type), and
   *   <li>we can successfully read the runtime info from the toolchain provider.
   * </ol>
   */
  @Nullable private final PyRuntimeInfo runtimeFromToolchain;

  /**
   * Symlink map from root-relative paths to 2to3 converted source artifacts.
   *
   * <p>Null if no 2to3 conversion is required.
   */
  @Nullable private final Map<PathFragment, Artifact> convertedFiles;

  private Artifact executable = null;

  private NestedSet<Artifact> filesToBuild = null;

  private static String getOrderErrorMessage(String fieldName, Order expected, Order actual) {
    return String.format(
        "Incompatible order for %s: expected 'default' or '%s', got '%s'",
        fieldName, expected.getStarlarkName(), actual.getStarlarkName());
  }

  // TODO(bazel-team): validatePackageAndSources is the result of refactoring while preserving
  // legacy behavior across some (but not all) Google-internal uses of PyCommon. Ideally all call
  // sites should be updated to expect the same validation steps.
  public PyCommon(
      RuleContext ruleContext, PythonSemantics semantics, boolean validatePackageAndSources) {
    this.ruleContext = ruleContext;
    this.semantics = semantics;
    this.version = ruleContext.getFragment(PythonConfiguration.class).getPythonVersion();
    this.sourcesVersion = initSrcsVersionAttr(ruleContext);
    this.dependencyTransitivePythonSources = initDependencyTransitivePythonSources(ruleContext);
    this.transitivePythonSources = initTransitivePythonSources(ruleContext);
    this.directPythonSources =
        initAndMaybeValidateDirectPythonSources(
            ruleContext, semantics, /*validate=*/ validatePackageAndSources);
    this.usesSharedLibraries = initUsesSharedLibraries(ruleContext);
    this.imports = initImports(ruleContext, semantics);
    this.hasPy2OnlySources = initHasPy2OnlySources(ruleContext, this.sourcesVersion);
    this.hasPy3OnlySources = initHasPy3OnlySources(ruleContext, this.sourcesVersion);
    this.runtimeFromToolchain = initRuntimeFromToolchain(ruleContext, this.version);
    this.convertedFiles = makeAndInitConvertedFiles(ruleContext, version, this.sourcesVersion);
    validatePythonVersionAttr();
    validateLegacyProviderNotUsedIfDisabled();
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
        ruleContext.getPrerequisiteArtifacts("srcs").filter(PyRuleClasses.PYTHON_SOURCE).list());
    return builder.build();
  }

  /**
   * Gathers transitive .py files from {@code deps} (not including this target's {@code srcs} and
   * adds them to {@code builder}.
   */
  private static void collectTransitivePythonSourcesFromDeps(
      RuleContext ruleContext, NestedSetBuilder<Artifact> builder) {
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps")) {
      try {
        builder.addTransitive(PyProviderUtils.getTransitiveSources(dep));
      } catch (EvalException e) {
        // Either the provider type or field type is bad.
        ruleContext.attributeError(
            "deps", String.format("In dep '%s': %s", dep.getLabel(), e.getMessage()));
      }
    }
  }

  private static List<Artifact> initAndMaybeValidateDirectPythonSources(
      RuleContext ruleContext, PythonSemantics semantics, boolean validate) {
    List<Artifact> sourceFiles = new ArrayList<>();
    // TODO(bazel-team): Need to get the transitive deps closure, not just the sources of the rule.
    for (TransitiveInfoCollection src :
        ruleContext.getPrerequisitesIf("srcs", FileProvider.class)) {
      // Make sure that none of the sources contain hyphens.
      if (validate
          && semantics.prohibitHyphensInPackagePaths()
          && Util.containsHyphen(src.getLabel().getPackageFragment())) {
        ruleContext.attributeError(
            "srcs", src.getLabel() + ": paths to Python packages may not contain '-'");
      }
      Iterable<Artifact> pySrcs =
          FileType.filter(
              src.getProvider(FileProvider.class).getFilesToBuild().toList(),
              PyRuleClasses.PYTHON_SOURCE);
      Iterables.addAll(sourceFiles, pySrcs);
      if (validate && Iterables.isEmpty(pySrcs)) {
        ruleContext.attributeWarning(
            "srcs", "rule '" + src.getLabel() + "' does not produce any Python source files");
      }
    }
    return sourceFiles;
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
              ruleContext.getPrerequisites("deps"), ruleContext.getPrerequisites("data"));
    } else {
      targets = ruleContext.getPrerequisites("deps");
    }
    for (TransitiveInfoCollection target : targets) {
      try {
        if (PyProviderUtils.getUsesSharedLibraries(target)) {
          return true;
        }
      } catch (EvalException e) {
        ruleContext.ruleError(String.format("In dep '%s': %s", target.getLabel(), e.getMessage()));
      }
    }
    return false;
  }

  private static NestedSet<String> initImports(RuleContext ruleContext, PythonSemantics semantics) {
    NestedSetBuilder<String> builder = NestedSetBuilder.compileOrder();
    builder.addAll(semantics.getImports(ruleContext));
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps")) {
      try {
        NestedSet<String> imports = PyProviderUtils.getImports(dep);
        if (!builder.getOrder().isCompatible(imports.getOrder())) {
          // TODO(brandjon): We should make order an invariant of the Python provider, and move this
          // check into PyInfo/PyStructUtils.
          ruleContext.ruleError(
              getOrderErrorMessage(PyStructUtils.IMPORTS, builder.getOrder(), imports.getOrder()));
        } else {
          builder.addTransitive(imports);
        }
      } catch (EvalException e) {
        ruleContext.attributeError(
            "deps", String.format("In dep '%s': %s", dep.getLabel(), e.getMessage()));
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
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps")) {
      try {
        if (PyProviderUtils.getHasPy2OnlySources(dep)) {
          return true;
        }
      } catch (EvalException e) {
        ruleContext.attributeError(
            "deps", String.format("In dep '%s': %s", dep.getLabel(), e.getMessage()));
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
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps")) {
      try {
        if (PyProviderUtils.getHasPy3OnlySources(dep)) {
          return true;
        }
      } catch (EvalException e) {
        ruleContext.attributeError(
            "deps", String.format("In dep '%s': %s", dep.getLabel(), e.getMessage()));
      }
    }
    return false;
  }

  /**
   * Retrieves the {@link PyRuntimeInfo} object in the given field of the given {@link
   * ToolchainInfo}.
   *
   * <p>If the field holds {@code None}, null is returned instead.
   *
   * <p>If the field does not exist on the given {@code ToolchainInfo}, or is not a {@code
   * PyRuntimeInfo} and not {@code None}, an error is reported on the {@code ruleContext} and null
   * is returned.
   *
   * <p>If the {@code PyRuntimeInfo} does not have {@code expectedVersion} as its Python version, an
   * error is reported on the {@code ruleContext} (but the provider is still returned).
   */
  @Nullable
  private static PyRuntimeInfo parseRuntimeField(
      RuleContext ruleContext,
      PythonVersion expectedVersion,
      ToolchainInfo toolchainInfo,
      String field) {
    Object fieldValue;
    try {
      fieldValue = toolchainInfo.getValue(field);
    } catch (EvalException e) {
      ruleContext.ruleError(
          String.format(
              "Error parsing the Python toolchain's ToolchainInfo: Could not retrieve field "
                  + "'%s': %s",
              field, e.getMessage()));
      return null;
    }
    if (fieldValue == null) {
      ruleContext.ruleError(
          String.format(
              "Error parsing the Python toolchain's ToolchainInfo: field '%s' is missing", field));
      return null;
    }
    if (fieldValue == NONE) {
      return null;
    }
    if (!(fieldValue instanceof PyRuntimeInfo)) {
      ruleContext.ruleError(
          String.format(
              "Error parsing the Python toolchain's ToolchainInfo: Expected a PyRuntimeInfo in "
                  + "field '%s', but got '%s'",
              field, Starlark.type(fieldValue)));
      return null;
    }
    PyRuntimeInfo pyRuntimeInfo = (PyRuntimeInfo) fieldValue;
    if (pyRuntimeInfo.getPythonVersion() != expectedVersion) {
      ruleContext.ruleError(
          String.format(
              "Error retrieving the Python runtime from the toolchain: Expected field '%s' to have "
                  + "a runtime with python_version = '%s', but got python_version = '%s'",
              field, expectedVersion.name(), pyRuntimeInfo.getPythonVersion().name()));
    }
    return pyRuntimeInfo;
  }

  /**
   * Returns a {@link PyRuntimeInfo} representing the runtime to use for this target, as retrieved
   * from the resolved Python toolchain.
   *
   * <p>If the configuration says to use the legacy mechanism for obtaining the runtime rather than
   * the toolchain mechanism, OR if this target's rule class does not define the
   * "$py_toolchain_type" attribute, then null is returned. In this case no attempt is made to
   * retrieve any toolchain information, and no errors are reported.
   *
   * <p>Otherwise, the toolchain provider structure is retrieved and validated, and any errors are
   * reported on the rule context. If we're unable to determine the runtime due to an error, or if
   * the toolchain does not specify a runtime for the version of Python we need, null is returned.
   *
   * @throws IllegalArgumentException if the rule class defines the "$py_toolchain_type" attribute
   *     but does not declare a requirement on the toolchain type
   */
  @Nullable
  private static PyRuntimeInfo initRuntimeFromToolchain(
      RuleContext ruleContext, PythonVersion version) {
    if (!shouldGetRuntimeFromToolchain(ruleContext)
        || !ruleContext.attributes().has("$py_toolchain_type", BuildType.NODEP_LABEL)) {
      return null;
    }
    Label toolchainType = ruleContext.attributes().get("$py_toolchain_type", BuildType.NODEP_LABEL);
    ToolchainInfo toolchainInfo = ruleContext.getToolchainContext().forToolchainType(toolchainType);
    Preconditions.checkArgument(
        toolchainInfo != null,
        "Could not retrieve a Python toolchain for '%s' rule",
        ruleContext.getRule().getRuleClass());

    PyRuntimeInfo py2RuntimeInfo =
        parseRuntimeField(ruleContext, PythonVersion.PY2, toolchainInfo, "py2_runtime");
    PyRuntimeInfo py3RuntimeInfo =
        parseRuntimeField(ruleContext, PythonVersion.PY3, toolchainInfo, "py3_runtime");
    Preconditions.checkState(version == PythonVersion.PY2 || version == PythonVersion.PY3);
    PyRuntimeInfo result = version == PythonVersion.PY2 ? py2RuntimeInfo : py3RuntimeInfo;
    if (result == null) {
      ruleContext.ruleError(
          String.format(
              "The Python toolchain does not provide a runtime for Python version %s",
              version.name()));
    }

    // Hack around the fact that the autodetecting Python toolchain, which is automatically
    // registered, does not yet support windows. In this case, we want to return null so that
    // BazelPythonSemantics falls back on --python_path. See toolchain.bzl.
    // TODO(#7844): Remove this hack when the autodetecting toolchain has a windows implementation.
    if (py2RuntimeInfo != null
        && py2RuntimeInfo.getInterpreterPathString() != null
        && py2RuntimeInfo
            .getInterpreterPathString()
            .equals("/_magic_pyruntime_sentinel_do_not_use")) {
      return null;
    }

    return result;
  }

  /**
   * If 2to3 conversion is to be done, creates the 2to3 actions and returns the map of converted
   * files; otherwise returns null.
   *
   * <p>May also return null and report a rule error if there is a problem creating an output file
   * for 2to3 conversion.
   */
  // TODO(#1393): 2to3 conversion doesn't work in Bazel and the attempt to invoke it for Bazel
  // should be removed / factored away into PythonSemantics.
  @Nullable
  private static Map<PathFragment, Artifact> makeAndInitConvertedFiles(
      RuleContext ruleContext, PythonVersion version, PythonVersion sourcesVersion) {
    if (sourcesVersion == PythonVersion.PY2 && version == PythonVersion.PY3) {
      Iterable<Artifact> artifacts =
          ruleContext.getPrerequisiteArtifacts("srcs").filter(PyRuleClasses.PYTHON_SOURCE).list();
      return PythonUtils.generate2to3Actions(ruleContext, artifacts);
    } else {
      return null;
    }
  }

  /**
   * Reports an attribute error if {@code python_version} cannot be parsed as {@code PY2}, {@code
   * PY3}, or the sentinel value.
   *
   * <p>This *should* be enforced by rule attribute validation ({@link
   * Attribute.Builder.allowedValues}), but this check is here to fail-fast just in case.
   */
  private void validatePythonVersionAttr() {
    AttributeMap attrs = ruleContext.attributes();
    if (!attrs.has(PYTHON_VERSION_ATTRIBUTE, Type.STRING)) {
      return;
    }
    String attrValue = attrs.get(PYTHON_VERSION_ATTRIBUTE, Type.STRING);
    try {
      PythonVersion.parseTargetOrSentinelValue(attrValue);
    } catch (IllegalArgumentException ex) {
      ruleContext.attributeError(
          PYTHON_VERSION_ATTRIBUTE,
          String.format("'%s' is not a valid value. Expected either 'PY2' or 'PY3'", attrValue));
    }
  }

  /**
   * Reports an attribute error if a target in {@code deps} passes the legacy "py" provider but this
   * is disallowed by the configuration.
   */
  private void validateLegacyProviderNotUsedIfDisabled() {
    if (!ruleContext.getFragment(PythonConfiguration.class).disallowLegacyPyProvider()) {
      return;
    }
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites("deps")) {
      if (PyProviderUtils.hasLegacyProvider(dep)) {
        ruleContext.attributeError(
            "deps",
            String.format(
                "In dep '%s': The legacy 'py' provider is disallowed. Migrate to the PyInfo "
                    + "provider instead. You can temporarily disable this failure with "
                    + "--incompatible_disallow_legacy_py_provider=false.",
                dep.getLabel()));
      }
    }
  }

  /**
   * If the Python version (as determined by the configuration) is inconsistent with {@link
   * #hasPy2OnlySources} or {@link #hasPy3OnlySources}, emits a {@link FailAction} that "generates"
   * the executable.
   *
   * <p>We use a {@code FailAction} rather than a rule error because we want to defer the error
   * until the execution phase. This way, we still get a configured target that the user can query
   * over with an aspect to find the exact transitive dependency that introduced the offending
   * version constraint. (See {@code <tools repo>//tools/python/srcs_version.bzl%find_requirements})
   *
   * @return true if a {@link FailAction} was created
   */
  private boolean maybeCreateFailActionDueToTransitiveSourcesVersion() {
    String errorTemplate =
        ruleContext.getLabel()
            + ": "
            + "This target is being built for Python %s but (transitively) includes Python %s-only "
            + "sources. You can get diagnostic information about which dependencies introduce this "
            + "version requirement by running the `find_requirements` aspect. If this is used in a "
            + "genrule, you may need to migrate from tools to exec_tools. For more info see "
            + "the documentation for the `srcs_version` attribute: "
            + semantics.getSrcsVersionDocURL();

    String error = null;
    if (version == PythonVersion.PY2 && hasPy3OnlySources) {
      error = String.format(errorTemplate, "2", "3");
    } else if (version == PythonVersion.PY3 && hasPy2OnlySources) {
      error = String.format(errorTemplate, "3", "2");
    }
    if (error == null) {
      return false;
    } else {
      ruleContext.registerAction(
          new FailAction(
              ruleContext.getActionOwner(),
              ImmutableList.of(executable),
              error,
              Code.INCORRECT_PYTHON_VERSION));
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
   * Returns whether, in the case that a user Python program fails, the stub script should emit a
   * warning that the failure may have been caused by the host configuration using the wrong Python
   * version.
   *
   * <p>This method should only be called for executable Python rules.
   *
   * <p>Background: Historically, Bazel did not necessarily launch a Python interpreter whose
   * version corresponded to the one determined by the analysis phase (#4815). Enabling Python
   * toolchains fixed this bug. However, this caused some builds to break due to targets that
   * contained Python-2-only code yet got analyzed for (and now run with) Python 3. This is
   * particularly problematic for the host configuration, where the value of {@code
   * --host_force_python} overrides the declared or implicit Python version of the target.
   *
   * <p>Our mitigation for this is to warn users when a Python target has a non-zero exit code and
   * the failure could be due to a bad Python version in the host configuration. In this case,
   * instead of just giving the user a confusing traceback of a PY2 vs PY3 error, we append a
   * diagnostic message to stderr. See #7899 and especially #8549 for context.
   *
   * <p>This method returns true when all of the following hold:
   *
   * <ol>
   *   <li>Python toolchains are enabled. (The warning is needed the most when toolchains are
   *       enabled, since that's an incompatible change likely to cause breakages. At the same time,
   *       warning when toolchains are disabled could be misleading, since we don't actually know
   *       whether the interpreter invoked at runtime is correct.)
   *   <li>The target is built in the host configuration. This avoids polluting stderr with spurious
   *       warnings for non-host-configured targets, while covering the most problematic case.
   *   <li>Either the value of {@code --host_force_python} overrode the target's normal Python
   *       version to a different value (in which case we know a mismatch occurred), or else {@code
   *       --host_force_python} is in agreement with the target's version but the target's version
   *       was set by default instead of explicitly (in which case we suspect the target may have
   *       been defined incorrectly).
   * </ol>
   *
   * @throws IllegalArgumentException if there is a problem parsing the Python version from the
   *     attributes; see {@link #readPythonVersionFromAttribute}.
   */
  // TODO(#6443): Remove this logic and the corresponding stub script logic once we no longer have
  // the possibility of Python binaries appearing in the host configuration.
  public boolean shouldWarnAboutHostVersionUponFailure() {
    // Only warn when toolchains are used.
    PythonConfiguration config = ruleContext.getFragment(PythonConfiguration.class);
    if (!config.useToolchains()) {
      return false;
    }
    // Only warn in the host config.
    if (!ruleContext.getConfiguration().isHostConfiguration()) {
      return false;
    }

    PythonVersion configVersion = config.getPythonVersion();
    PythonVersion attrVersion = readPythonVersionFromAttribute(ruleContext.attributes());
    if (attrVersion == null) {
      // Warn if the version wasn't set explicitly.
      return true;
    } else {
      // Warn if the explicit version is different from the host config's version.
      return configVersion != attrVersion;
    }
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

  /**
   * Returns {@code true} if the Python runtime should be obtained from the Python toolchain (as per
   * {@code --incompatible_use_python_toolchains}), as opposed to through the legacy mechanism
   * specified in the {@link PythonSemantics} (e.g., {@code --python_top}).
   */
  public boolean shouldGetRuntimeFromToolchain() {
    return shouldGetRuntimeFromToolchain(ruleContext);
  }

  private static boolean shouldGetRuntimeFromToolchain(RuleContext ruleContext) {
    return ruleContext.getFragment(PythonConfiguration.class).useToolchains();
  }

  /**
   * Returns a {@link PyRuntimeInfo} representing the runtime to use for this target, as retrieved
   * from the resolved toolchain.
   *
   * <p>This may only be called for executable Python rules (rules defining the attribute
   * "$py_toolchain_type", i.e. {@code py_binary} and {@code py_test}). In addition, it may not be
   * called if {@link #shouldGetRuntimeFromToolchain()} returns false.
   *
   * <p>If there was a problem retrieving the runtime information from the toolchain, null is
   * returned. An error would have already been reported on the rule context at {@code PyCommon}
   * initialization time.
   */
  @Nullable
  public PyRuntimeInfo getRuntimeFromToolchain() {
    Preconditions.checkArgument(
        ruleContext.attributes().has("$py_toolchain_type", BuildType.NODEP_LABEL),
        "Cannot retrieve Python toolchain information for '%s' rule",
        ruleContext.getRule().getRuleClass());
    Preconditions.checkArgument(
        shouldGetRuntimeFromToolchain(),
        "Access to the Python toolchain is disabled by --incompatible_use_python_toolchains=false");

    return runtimeFromToolchain;
  }

  public Map<PathFragment, Artifact> getConvertedFiles() {
    return convertedFiles;
  }

  public void initBinary(List<Artifact> srcs) {
    Preconditions.checkNotNull(version);

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
      filesToBuildBuilder.add(getPythonStubArtifactForWindows(executable));
    }

    filesToBuild = filesToBuildBuilder.build();

    if (ruleContext.hasErrors()) {
      return;
    }

    addPyExtraActionPseudoAction();
  }

  /** @return an artifact next to the executable file with a given suffix. */
  private Artifact getArtifactWithExtension(Artifact executable, String extension) {
    // On Windows, the Python executable has .exe extension on Windows,
    // On Linux, the Python executable has no extension.
    // We can't use ruleContext#getRelatedArtifact because it would mangle files with dots in the
    // name on non-Windows platforms.
    PathFragment pathFragment =
        executable.getOutputDirRelativePath(
            ruleContext.getConfiguration().isSiblingRepositoryLayout());
    String fileName = executable.getFilename();
    if (OS.getCurrent() == OS.WINDOWS) {
      Preconditions.checkArgument(fileName.endsWith(".exe"));
      fileName = fileName.substring(0, fileName.length() - 4) + extension;
    } else {
      fileName = fileName + extension;
    }
    return ruleContext.getDerivedArtifact(pathFragment.replaceName(fileName), executable.getRoot());
  }

  /** Returns an artifact next to the executable file with ".zip" suffix. */
  public Artifact getPythonZipArtifact(Artifact executable) {
    return getArtifactWithExtension(executable, ".zip");
  }

  /**
   * Returns an artifact next to the executable file with ".temp" suffix. Used only if we're
   * building a zip.
   */
  public Artifact getPythonIntermediateStubArtifact(Artifact executable) {
    return getArtifactWithExtension(executable, ".temp");
  }

  /** Returns an artifact next to the executable file with no suffix. Only called for Windows. */
  public Artifact getPythonStubArtifactForWindows(Artifact executable) {
    return ruleContext.getRelatedArtifact(
        executable.getOutputDirRelativePath(
            ruleContext.getConfiguration().isSiblingRepositoryLayout()),
        "");
  }

  /**
   * Adds a PyInfo or legacy "py" provider.
   *
   * <p>This is a public method because some rules just want a PyInfo provider without the other
   * things py_library needs.
   */
  public void addPyInfoProvider(RuleConfiguredTargetBuilder builder) {
    boolean createLegacyPyProvider =
        !ruleContext.getFragment(PythonConfiguration.class).disallowLegacyPyProvider();
    PyProviderUtils.builder(createLegacyPyProvider)
        .setTransitiveSources(transitivePythonSources)
        .setUsesSharedLibraries(usesSharedLibraries)
        .setImports(imports)
        .setHasPy2OnlySources(hasPy2OnlySources)
        .setHasPy3OnlySources(hasPy3OnlySources)
        .buildAndAddToTarget(builder);
  }

  public void addCommonTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder, NestedSet<Artifact> filesToBuild) {
    addPyInfoProvider(builder);

    // Add PyRuntimeInfo if this is an executable rule.
    if (runtimeFromToolchain != null) {
      builder.addNativeDeclaredProvider(runtimeFromToolchain);
    }

    builder
        .addNativeDeclaredProvider(
            InstrumentedFilesCollector.collect(
                ruleContext, semantics.getCoverageInstrumentationSpec()))
        // Python targets are not really compilable. The best we can do is make sure that all
        // generated source files are ready.
        .addOutputGroup(OutputGroupInfo.FILES_TO_COMPILE, transitivePythonSources)
        .addOutputGroup(OutputGroupInfo.COMPILATION_PREREQUISITES, transitivePythonSources);
  }

  /** Returns a list of the source artifacts */
  public List<Artifact> getPythonSources() {
    return convertedFiles != null
        ? ImmutableList.copyOf(convertedFiles.values())
        : directPythonSources;
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
            ruleContext.getPrerequisiteArtifacts("srcs").list(),
            // We must not add the files declared in the srcs of this rule.;
            dependencyTransitivePythonSources,
            PseudoAction.getDummyOutput(ruleContext)));
  }

  /**
   * Creates a {@link PseudoAction} that is only used for providing information to the blaze
   * extra_action feature.
   */
  public static Action makePyExtraActionPseudoAction(
      ActionOwner owner,
      List<Artifact> sources,
      NestedSet<Artifact> dependencies,
      Artifact output) {

    PythonInfo info =
        PythonInfo.newBuilder()
            .addAllSourceFile(Artifact.toExecPaths(sources))
            .addAllDepFile(Artifact.toExecPaths(dependencies.toList()))
            .build();

    return new PyPseudoAction(
        owner,
        NestedSetBuilder.<Artifact>stableOrder()
            .addAll(sources)
            .addTransitive(dependencies)
            .build(),
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
    for (Artifact outItem : ruleContext.getPrerequisiteArtifacts("srcs").list()) {
      if (outItem.getRootRelativePath().endsWith(mainSourcePath)) {
        if (mainArtifact == null) {
          mainArtifact = outItem;
        } else {
          ruleContext.attributeError(
              "srcs",
              buildMultipleMainMatchesErrorText(
                  explicitMain,
                  mainSourceName,
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
  public void createExecutable(CcInfo ccInfo, Runfiles.Builder defaultRunfilesBuilder)
      throws InterruptedException, RuleErrorException {
    boolean failed = maybeCreateFailActionDueToTransitiveSourcesVersion();
    if (!failed) {
      semantics.createExecutable(ruleContext, this, ccInfo, defaultRunfilesBuilder);
    }
  }

  private static String buildMultipleMainMatchesErrorText(
      boolean explicit, String proposedMainName, String match1, String match2) {
    String errorText;
    if (explicit) {
      errorText =
          String.format(
              "file name '%s' specified by 'main' attribute matches multiple files: e.g., '%s' and"
                  + " '%s'",
              proposedMainName, match1, match2);
    } else {
      errorText =
          String.format(
              "default main file name '%s' matches multiple files.  Perhaps specify an explicit"
                  + " file with 'main' attribute?  Matches were: '%s' and '%s'",
              proposedMainName, match1, match2);
    }
    return errorText;
  }

  private static String buildNoMainMatchesErrorText(boolean explicit, String proposedMainName) {
    String errorText;
    if (explicit) {
      errorText = "could not find '" + proposedMainName + "' as specified by 'main' attribute";
    } else {
      errorText =
          String.format(
              "corresponding default '%s' does not appear in srcs. Add it or override default file"
                  + " name with a 'main' attribute",
              proposedMainName);
    }
    return errorText;
  }

  // Used purely to set the legacy ActionType of the ExtraActionInfo.
  @Immutable
  private static final class PyPseudoAction extends PseudoAction<PythonInfo> {
    private static final UUID ACTION_UUID = UUID.fromString("8d720129-bc1a-481f-8c4c-dbe11dcef319");

    public PyPseudoAction(
        ActionOwner owner,
        NestedSet<Artifact> inputs,
        Collection<Artifact> outputs,
        String mnemonic,
        GeneratedExtension<ExtraActionInfo, PythonInfo> infoExtension,
        PythonInfo info) {
      super(ACTION_UUID, owner, inputs, outputs, mnemonic, infoExtension, info);
    }
  }
}
