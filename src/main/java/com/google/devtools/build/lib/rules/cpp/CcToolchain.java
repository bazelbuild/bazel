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

import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.CompilationHelper;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.cpp.FdoSupport.FdoException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Implementation for the cc_toolchain rule.
 */
public class CcToolchain implements RuleConfiguredTargetFactory {

  /**
   * This file (found under the sysroot) may be unconditionally included in every C/C++ compilation.
   */
  private static final PathFragment BUILTIN_INCLUDE_FILE_SUFFIX =
      new PathFragment("include/stdc-predef.h");

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    TransitiveInfoCollection lipoContextCollector =
        ruleContext.getPrerequisite(":lipo_context_collector", Mode.DONT_CHECK);
    if (lipoContextCollector != null
        && lipoContextCollector.getProvider(LipoContextProvider.class) == null) {
      ruleContext.ruleError("--lipo_context must point to a cc_binary or a cc_test rule");
      return null;
    }

    CppConfiguration cppConfiguration =
        Preconditions.checkNotNull(ruleContext.getFragment(CppConfiguration.class));
    Path fdoZip = ruleContext.getConfiguration().getCompilationMode() == CompilationMode.OPT
        ? cppConfiguration.getFdoZip()
        : null;
    SkyKey fdoKey = FdoSupportValue.key(
        cppConfiguration.getLipoMode(),
        fdoZip,
        cppConfiguration.getFdoInstrument());

    SkyFunction.Environment skyframeEnv = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    FdoSupportValue fdoSupport;
    try {
      fdoSupport = (FdoSupportValue) skyframeEnv.getValueOrThrow(
          fdoKey, FdoException.class, IOException.class);
    } catch (FdoException | IOException e) {
      ruleContext.ruleError("cannot initialize FDO: " + e.getMessage());
      return null;
    }

    if (skyframeEnv.valuesMissing()) {
      return null;
    }

    final Label label = ruleContext.getLabel();
    final NestedSet<Artifact> crosstool = ruleContext.getPrerequisite("all_files", Mode.HOST)
        .getProvider(FileProvider.class).getFilesToBuild();
    final NestedSet<Artifact> crosstoolMiddleman = getFiles(ruleContext, "all_files");
    final NestedSet<Artifact> compile = getFiles(ruleContext, "compiler_files");
    final NestedSet<Artifact> strip = getFiles(ruleContext, "strip_files");
    final NestedSet<Artifact> objcopy = getFiles(ruleContext, "objcopy_files");
    final NestedSet<Artifact> link = getFiles(ruleContext, "linker_files");
    final NestedSet<Artifact> dwp = getFiles(ruleContext, "dwp_files");
    final NestedSet<Artifact> libcLink = inputsForLibc(ruleContext);
    String purposePrefix = Actions.escapeLabel(label) + "_";
    String runtimeSolibDirBase = "_solib_" + "_" + Actions.escapeLabel(label);
    final PathFragment runtimeSolibDir = ruleContext.getConfiguration()
        .getBinFragment().getRelative(runtimeSolibDirBase);

    // Static runtime inputs.
    TransitiveInfoCollection staticRuntimeLibDep = selectDep(ruleContext, "static_runtime_libs",
        cppConfiguration.getStaticRuntimeLibsLabel());
    final NestedSet<Artifact> staticRuntimeLinkInputs;
    final Artifact staticRuntimeLinkMiddleman;
    if (cppConfiguration.supportsEmbeddedRuntimes()) {
      staticRuntimeLinkInputs = staticRuntimeLibDep
          .getProvider(FileProvider.class)
          .getFilesToBuild();
    } else {
      staticRuntimeLinkInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    if (!staticRuntimeLinkInputs.isEmpty()) {
      NestedSet<Artifact> staticRuntimeLinkMiddlemanSet = CompilationHelper.getAggregatingMiddleman(
          ruleContext,
          purposePrefix + "static_runtime_link",
          staticRuntimeLibDep);
      staticRuntimeLinkMiddleman = staticRuntimeLinkMiddlemanSet.isEmpty()
          ? null : Iterables.getOnlyElement(staticRuntimeLinkMiddlemanSet);
    } else {
      staticRuntimeLinkMiddleman = null;
    }

    Preconditions.checkState(
        (staticRuntimeLinkMiddleman == null) == staticRuntimeLinkInputs.isEmpty());

    // Dynamic runtime inputs.
    TransitiveInfoCollection dynamicRuntimeLibDep = selectDep(ruleContext, "dynamic_runtime_libs",
        cppConfiguration.getDynamicRuntimeLibsLabel());
    NestedSet<Artifact> dynamicRuntimeLinkSymlinks;
    List<Artifact> dynamicRuntimeLinkInputs = new ArrayList<>();
    Artifact dynamicRuntimeLinkMiddleman;
    if (cppConfiguration.supportsEmbeddedRuntimes()) {
      NestedSetBuilder<Artifact> dynamicRuntimeLinkSymlinksBuilder = NestedSetBuilder.stableOrder();
      for (Artifact artifact : dynamicRuntimeLibDep
          .getProvider(FileProvider.class).getFilesToBuild()) {
        if (CppHelper.SHARED_LIBRARY_FILETYPES.matches(artifact.getFilename())) {
          dynamicRuntimeLinkInputs.add(artifact);
          dynamicRuntimeLinkSymlinksBuilder.add(SolibSymlinkAction.getCppRuntimeSymlink(
              ruleContext, artifact, runtimeSolibDirBase,
              ruleContext.getConfiguration()));
        }
      }
      dynamicRuntimeLinkSymlinks = dynamicRuntimeLinkSymlinksBuilder.build();
    } else {
      dynamicRuntimeLinkSymlinks = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    if (!dynamicRuntimeLinkInputs.isEmpty()) {
      List<Artifact> dynamicRuntimeLinkMiddlemanSet =
          CppHelper.getAggregatingMiddlemanForCppRuntimes(
              ruleContext,
              purposePrefix + "dynamic_runtime_link",
              dynamicRuntimeLinkInputs,
              runtimeSolibDirBase,
              ruleContext.getConfiguration());
      dynamicRuntimeLinkMiddleman = dynamicRuntimeLinkMiddlemanSet.isEmpty()
          ? null : Iterables.getOnlyElement(dynamicRuntimeLinkMiddlemanSet);
    } else {
      dynamicRuntimeLinkMiddleman = null;
    }

    Preconditions.checkState(
        (dynamicRuntimeLinkMiddleman == null) == dynamicRuntimeLinkSymlinks.isEmpty());

    CppCompilationContext.Builder contextBuilder =
        new CppCompilationContext.Builder(ruleContext);
    CppModuleMap moduleMap = createCrosstoolModuleMap(ruleContext);
    if (moduleMap != null) {
      contextBuilder.setCppModuleMap(moduleMap);
    }
    final CppCompilationContext context = contextBuilder.build();
    boolean supportsParamFiles = ruleContext.attributes().get("supports_param_files", BOOLEAN);
    boolean supportsHeaderParsing =
        ruleContext.attributes().get("supports_header_parsing", BOOLEAN);

    NestedSetBuilder<Pair<String, String>> coverageEnvironment = NestedSetBuilder.compileOrder();

    coverageEnvironment.add(Pair.of(
        "COVERAGE_GCOV_PATH", cppConfiguration.getGcovExecutable().getPathString()));
    if (cppConfiguration.getFdoInstrument() != null) {
      coverageEnvironment.add(Pair.of(
          "FDO_DIR", cppConfiguration.getFdoInstrument().getPathString()));
    }

    NestedSet<Artifact> coverage = getOptionalFiles(ruleContext, "coverage_files");
    if (coverage.isEmpty()) {
      coverage = crosstool;
    }

    CcToolchainProvider provider =
        new CcToolchainProvider(
            cppConfiguration,
            crosstool,
            fullInputsForCrosstool(ruleContext, crosstoolMiddleman),
            compile,
            strip,
            objcopy,
            fullInputsForLink(ruleContext, link),
            ruleContext.getPrerequisiteArtifact("$interface_library_builder", Mode.HOST),
            dwp,
            coverage,
            libcLink,
            staticRuntimeLinkInputs,
            staticRuntimeLinkMiddleman,
            dynamicRuntimeLinkSymlinks,
            dynamicRuntimeLinkMiddleman,
            runtimeSolibDir,
            context,
            supportsParamFiles,
            supportsHeaderParsing,
            getBuildVariables(ruleContext),
            getBuiltinIncludes(ruleContext),
            coverageEnvironment.build(),
            ruleContext.getPrerequisiteArtifact("$link_dynamic_library_tool", Mode.HOST),
            getEnvironment(ruleContext));
    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .add(CcToolchainProvider.class, provider)
            .add(FdoSupportProvider.class,
                fdoSupport.getFdoSupport().createFdoSupportProvider(ruleContext))
            .setFilesToBuild(new NestedSetBuilder<Artifact>(Order.STABLE_ORDER).build())
            .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));

    // If output_license is specified on the cc_toolchain rule, override the transitive licenses
    // with that one. This is necessary because cc_toolchain is used in the target configuration,
    // but it is sort-of-kind-of a tool, but various parts of it are linked into the output...
    // ...so we trust the judgment of the author of the cc_toolchain rule to figure out what
    // licenses should be propagated to C++ targets.
    // TODO(elenairina): Remove this and use Attribute.Builder.useOutputLicenses() on the
    // :cc_toolchain attribute instead.
    final License outputLicense =
        ruleContext.getRule().getToolOutputLicense(ruleContext.attributes());
    if (outputLicense != null && !outputLicense.equals(License.NO_LICENSE)) {
      final NestedSet<TargetLicense> license = NestedSetBuilder.create(Order.STABLE_ORDER,
          new TargetLicense(ruleContext.getLabel(), outputLicense));
      LicensesProvider licensesProvider = new LicensesProvider() {
        @Override
        public NestedSet<TargetLicense> getTransitiveLicenses() {
          return license;
        }

        @Override
        public TargetLicense getOutputLicenses() {
          return new TargetLicense(label, outputLicense);
        }

        @Override
        public boolean hasOutputLicenses() {
          return true;
        }

      };

      builder.add(LicensesProvider.class, licensesProvider);
    }

    return builder.build();
  }

  private ImmutableList<Artifact> getBuiltinIncludes(RuleContext ruleContext) {
    ImmutableList.Builder<Artifact> result = ImmutableList.builder();
    for (Artifact artifact : inputsForLibc(ruleContext)) {
      if (artifact.getExecPath().endsWith(BUILTIN_INCLUDE_FILE_SUFFIX)) {
        result.add(artifact);
      }
    }

    return result.build();
  }

  private NestedSet<Artifact> inputsForLibc(RuleContext ruleContext) {
    TransitiveInfoCollection libc = ruleContext.getPrerequisite(":libc_top", Mode.HOST);
    return libc != null
        ? libc.getProvider(FileProvider.class).getFilesToBuild()
        : NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER);
  }

  private NestedSet<Artifact> fullInputsForCrosstool(RuleContext ruleContext,
      NestedSet<Artifact> crosstoolMiddleman) {
    return NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(crosstoolMiddleman)
        .addTransitive(AnalysisUtils.getMiddlemanFor(ruleContext, ":libc_top"))
        .build();
  }

  /**
   * Returns the crosstool-derived link action inputs for a given rule. Adds the given set of
   * artifacts as extra inputs.
   */
  protected NestedSet<Artifact> fullInputsForLink(
      RuleContext ruleContext, NestedSet<Artifact> link) {
    return NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(link)
        .addTransitive(AnalysisUtils.getMiddlemanFor(ruleContext, ":libc_top"))
        .add(ruleContext.getPrerequisiteArtifact("$interface_library_builder", Mode.HOST))
        .add(ruleContext.getPrerequisiteArtifact("$link_dynamic_library_tool", Mode.HOST))
        .build();
  }

  private CppModuleMap createCrosstoolModuleMap(RuleContext ruleContext) {
    if (ruleContext.getPrerequisite("module_map", Mode.HOST) == null) {
      return null;
    }
    Artifact moduleMapArtifact = ruleContext.getPrerequisiteArtifact("module_map", Mode.HOST);
    if (moduleMapArtifact == null) {
      return null;
    }
    return new CppModuleMap(moduleMapArtifact, "crosstool");
  }

  private TransitiveInfoCollection selectDep(
      RuleContext ruleContext, String attribute, Label label) {
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisites(attribute, Mode.TARGET)) {
      if (dep.getLabel().equals(label)) {
        return dep;
      }
    }

    return ruleContext.getPrerequisites(attribute, Mode.TARGET).get(0);
  }

  private NestedSet<Artifact> getFiles(RuleContext context, String attribute) {
    TransitiveInfoCollection dep = context.getPrerequisite(attribute, Mode.HOST);
    MiddlemanProvider middlemanProvider = dep.getProvider(MiddlemanProvider.class);
    // We use the middleman if we can (if the dep is a filegroup), otherwise, just the regular
    // filesToBuild (e.g. if it is a simple input file)
    return middlemanProvider != null
        ? middlemanProvider.getMiddlemanArtifact()
        : dep.getProvider(FileProvider.class).getFilesToBuild();
  }

  private NestedSet<Artifact> getOptionalFiles(RuleContext context, String attribute) {
    TransitiveInfoCollection dep = context.getPrerequisite(attribute, Mode.HOST);
    return dep != null
        ? getFiles(context, attribute)
        : NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER);
  }

  /**
   * Returns a map that should be templated into the crosstool as build variables
   *
   * @param ruleContext the rule context
   * @throws RuleErrorException if there are configuration errors making it impossible to resolve
   *     certain build variables of this toolchain
   */
  protected final Map<String, String> getBuildVariables(RuleContext ruleContext)
      throws RuleErrorException {
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    ImmutableMap.Builder<String, String> variables = ImmutableMap.builder();
    if (cppConfiguration.getSysroot() != null) {
      variables.put(
          CppRuleClasses.SYSROOT_VARIABLE, cppConfiguration.getSysroot().getSafePathString());
    }
    variables.putAll(getLocalBuildVariables(ruleContext));
    return variables.build();
  }

  /**
   * Returns a map that should be templated into the crosstool as build variables. Is meant to be
   * overridden by subclasses of CcToolchain.
   *
   * @param ruleContext the rule context
   * @throws RuleErrorException if there are configuration errors making it impossible to resolve
   *     certain build variables of this toolchain
   */
  protected Map<String, String> getLocalBuildVariables(RuleContext ruleContext)
      throws RuleErrorException {
    return ImmutableMap.<String, String>of();
  }

  /**
   * Returns a map of environment variables to be added to the compile actions created for this
   * toolchain. Ideally, this will get replaced by features, which also allow setting env variables.
   *
   * @param ruleContext the rule context
   */
  protected ImmutableMap<String, String> getEnvironment(RuleContext ruleContext) {
    return ImmutableMap.<String, String>of();
  }
}
