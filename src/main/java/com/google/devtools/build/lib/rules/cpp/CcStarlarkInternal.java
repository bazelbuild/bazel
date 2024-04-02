// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.analysis.LicensesProviderImpl;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.Linkstamp;
import com.google.devtools.build.lib.starlarkbuildapi.NativeComputedDefaultApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.Location;

/** Utility methods for rules in Starlark Builtins */
@StarlarkBuiltin(name = "cc_internal", category = DocCategory.BUILTIN, documented = false)
public class CcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "cc_internal";

  @Nullable
  private static <T> T nullIfNone(Object object, Class<T> type) {
    return object != Starlark.NONE ? type.cast(object) : null;
  }

  private ImmutableMap<String, PathFragment> castDict(Dict<?, ?> d) throws EvalException {
    return Dict.cast(d, String.class, String.class, "tool_paths").entrySet().stream()
        .map(p -> Pair.of(p.getKey(), PathFragment.create(p.getValue())))
        .collect(toImmutableMap(Pair::getFirst, Pair::getSecond));
  }

  @StarlarkMethod(
      name = "cc_toolchain_variables",
      documented = false,
      parameters = {
        @Param(name = "vars", positional = false, named = true),
      })
  public CcToolchainVariables getCcToolchainVariables(Object vars) throws EvalException {
    CcToolchainVariables.Builder ccToolchainVariables = CcToolchainVariables.builder();
    for (Map.Entry<String, String> entry :
        Dict.noneableCast(vars, String.class, String.class, "vars").entrySet()) {
      ccToolchainVariables.addStringVariable(entry.getKey(), entry.getValue());
    }
    return ccToolchainVariables.build();
  }

  @StarlarkMethod(
      name = "solib_symlink_action",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "artifact", positional = false, named = true),
        @Param(name = "solib_directory", positional = false, named = true),
        @Param(name = "runtime_solib_dir_base", positional = false, named = true),
      })
  public Artifact solibSymlinkAction(
      StarlarkRuleContext ruleContext,
      Artifact artifact,
      String solibDirectory,
      String runtimeSolibDirBase) {
    return SolibSymlinkAction.getCppRuntimeSymlink(
        ruleContext.getRuleContext(), artifact, solibDirectory, runtimeSolibDirBase);
  }

  @StarlarkMethod(
      name = "fdo_context",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "cpp_config", positional = false, named = true),
        @Param(name = "tool_paths", positional = false, named = true),
        @Param(name = "fdo_prefetch_provider", positional = false, named = true),
        @Param(name = "propeller_optimize_provider", positional = false, named = true),
        @Param(name = "mem_prof_profile_provider", positional = false, named = true),
        @Param(name = "fdo_optimize_provider", positional = false, named = true),
        @Param(name = "fdo_profile_provider", positional = false, named = true),
        @Param(name = "x_fdo_profile_provider", positional = false, named = true),
        @Param(name = "cs_fdo_profile_provider", positional = false, named = true),
        @Param(name = "all_files", positional = false, named = true),
        @Param(name = "zipper", positional = false, named = true),
        @Param(name = "cc_toolchain_config_info", positional = false, named = true),
        @Param(name = "fdo_optimize_artifacts", positional = false, named = true),
        @Param(name = "fdo_optimize_label", positional = false, named = true),
      },
      allowReturnNones = true)
  @Nullable
  public FdoContext fdoContext(
      StarlarkRuleContext ruleContext,
      BuildConfigurationValue configuration,
      CppConfiguration cppConfiguration,
      Dict<?, ?> toolPathsDict,
      Object fdoPrefetchProvider,
      Object propellerOptimizeProvider,
      Object memProfProfileProvider,
      Object fdoOptimizeProvider,
      Object fdoProfileProvider,
      Object xFdoProfileProvider,
      Object csFdoProfileProvider,
      Object allFilesObject,
      Object zipper,
      CcToolchainConfigInfo ccToolchainConfigInfo,
      Sequence<?> fdoOptimizeArtifacts,
      Object fdoOptimizeLabel)
      throws EvalException, InterruptedException {
    NestedSet<Artifact> allFiles = null;

    try {
      allFiles = ((Depset) allFilesObject).getSet(Artifact.class);
    } catch (TypeException e) {
      throw new EvalException(e);
    }

    return FdoHelper.getFdoContext(
        ruleContext.getRuleContext(),
        configuration,
        cppConfiguration,
        castDict(toolPathsDict),
        nullIfNone(fdoPrefetchProvider, StructImpl.class),
        nullIfNone(propellerOptimizeProvider, StructImpl.class),
        nullIfNone(memProfProfileProvider, StructImpl.class),
        nullIfNone(fdoOptimizeProvider, StructImpl.class),
        nullIfNone(fdoProfileProvider, StructImpl.class),
        nullIfNone(xFdoProfileProvider, StructImpl.class),
        nullIfNone(csFdoProfileProvider, StructImpl.class),
        allFiles,
        nullIfNone(zipper, Artifact.class),
        ccToolchainConfigInfo,
        Sequence.cast(fdoOptimizeArtifacts, Artifact.class, "fdo_optimize_artifacts")
            .getImmutableList(),
        nullIfNone(fdoOptimizeLabel, Label.class));
  }

  @StarlarkMethod(
      name = "cc_toolchain_features",
      documented = false,
      parameters = {
        @Param(name = "toolchain_config_info", positional = false, named = true),
        @Param(name = "tools_directory", positional = false, named = true),
      })
  public CcToolchainFeatures ccToolchainFeatures(
      CcToolchainConfigInfo ccToolchainConfigInfo, String toolsDirectoryPathString)
      throws EvalException {
    return new CcToolchainFeatures(
        ccToolchainConfigInfo, PathFragment.create(toolsDirectoryPathString));
  }

  @StarlarkMethod(
      name = "is_package_headers_checking_mode_set",
      documented = false,
      parameters = {@Param(name = "ctx", positional = false, named = true)})
  public boolean isPackageHeadersCheckingModeSetForStarlark(
      StarlarkRuleContext starlarkRuleContext) {
    return starlarkRuleContext
        .getRuleContext()
        .getRule()
        .getPackage()
        .getPackageArgs()
        .isDefaultHdrsCheckSet();
  }

  @StarlarkMethod(
      name = "package_headers_checking_mode",
      documented = false,
      parameters = {@Param(name = "ctx", positional = false, named = true)})
  public String getPackageHeadersCheckingModeForStarlark(StarlarkRuleContext starlarkRuleContext) {
    return starlarkRuleContext
        .getRuleContext()
        .getRule()
        .getPackage()
        .getPackageArgs()
        .getDefaultHdrsCheck();
  }

  @StarlarkMethod(
      name = "is_package_headers_checking_mode_set_for_aspect",
      documented = false,
      parameters = {@Param(name = "ctx", positional = false, named = true)})
  public boolean isPackageHeadersCheckingModeSetForStarlarkAspect(
      StarlarkRuleContext starlarkRuleContext) {
    return starlarkRuleContext
        .getRuleContext()
        .getTarget()
        .getPackage()
        .getPackageArgs()
        .isDefaultHdrsCheckSet();
  }

  @StarlarkMethod(
      name = "package_headers_checking_mode_for_aspect",
      documented = false,
      parameters = {@Param(name = "ctx", positional = false, named = true)})
  public String getPackageHeadersCheckingModeForStarlarkAspect(
      StarlarkRuleContext starlarkRuleContext) {
    return starlarkRuleContext
        .getRuleContext()
        .getTarget()
        .getPackage()
        .getPackageArgs()
        .getDefaultHdrsCheck();
  }

  @StarlarkMethod(
      name = "create_common",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public CcCommon createCommon(StarlarkRuleContext starlarkRuleContext) {
    return new CcCommon(starlarkRuleContext.getRuleContext());
  }

  @StarlarkMethod(name = "launcher_provider", documented = false, structField = true)
  public ProviderApi getCcLauncherInfoProvider() throws EvalException {
    return CcLauncherInfo.PROVIDER;
  }

  @StarlarkMethod(
      name = "create_linkstamp",
      documented = false,
      parameters = {
        @Param(name = "actions", positional = false, named = true),
        @Param(name = "linkstamp", positional = false, named = true),
        @Param(name = "compilation_context", positional = false, named = true),
      })
  public Linkstamp createLinkstamp(
      StarlarkActionFactory starlarkActionFactoryApi,
      Artifact linkstamp,
      CcCompilationContext ccCompilationContext)
      throws EvalException {
    try {
      return new Linkstamp( // throws InterruptedException
          linkstamp,
          ccCompilationContext.getDeclaredIncludeSrcs(),
          starlarkActionFactoryApi.getRuleContext().getActionKeyContext());
    } catch (CommandLineExpansionException | InterruptedException ex) {
      throw new EvalException(ex);
    }
  }

  /**
   * TODO(bazel-team): This can be re-written directly to Starlark but it will cause a memory
   * regression due to the way StarlarkComputedDefault is stored for each rule.
   */
  static class StlComputedDefault extends ComputedDefault implements NativeComputedDefaultApi {
    @Override
    @Nullable
    public Object getDefault(AttributeMap rule) {
      return rule.getOrDefault("tags", Types.STRING_LIST, ImmutableList.of()).contains("__CC_STL__")
          ? null
          : Label.parseCanonicalUnchecked("@//third_party/stl");
    }
  }

  @StarlarkMethod(name = "stl_computed_default", documented = false)
  public ComputedDefault getStlComputedDefault() {
    return new StlComputedDefault();
  }

  @StarlarkMethod(
      name = "create_cc_launcher_info",
      doc = "Create a CcLauncherInfo instance.",
      parameters = {
        @Param(
            name = "cc_info",
            positional = false,
            named = true,
            doc = "CcInfo instance.",
            allowedTypes = {@ParamType(type = CcInfo.class)}),
        @Param(
            name = "compilation_outputs",
            positional = false,
            named = true,
            doc = "CcCompilationOutputs instance.",
            allowedTypes = {@ParamType(type = CcCompilationOutputs.class)})
      })
  public CcLauncherInfo createCcLauncherInfo(
      CcInfo ccInfo, CcCompilationOutputs compilationOutputs) {
    return new CcLauncherInfo(ccInfo, compilationOutputs);
  }

  private static final StarlarkProvider starlarkCcTestRunnerInfo =
      StarlarkProvider.builder(Location.BUILTIN)
          .setExported(
              new StarlarkProvider.Key(
                  Label.parseCanonicalUnchecked("//tools/cpp/cc_test:toolchain.bzl"),
                  "CcTestRunnerInfo"))
          .build();

  @StarlarkMethod(name = "CcTestRunnerInfo", documented = false, structField = true)
  public StarlarkProvider ccTestRunnerInfo() throws EvalException {
    return starlarkCcTestRunnerInfo;
  }

  // This looks ugly, however it is necessary. Good thing is we are planning to get rid of genfiles
  // directory altogether so this method has a bright future(of being removed).
  @StarlarkMethod(
      name = "bin_or_genfiles_relative_to_unique_directory",
      documented = false,
      parameters = {
        @Param(name = "actions", positional = false, named = true),
        @Param(name = "unique_directory", positional = false, named = true),
      })
  public String binOrGenfilesRelativeToUniqueDirectory(
      StarlarkActionFactory actions, String uniqueDirectory) {
    ActionConstructionContext actionConstructionContext = actions.getRuleContext();
    return actionConstructionContext
        .getBinOrGenfilesDirectory()
        .getExecPath()
        .getRelative(
            actionConstructionContext.getUniqueDirectory(PathFragment.create(uniqueDirectory)))
        .getPathString();
  }

  @StarlarkMethod(
      name = "create_umbrella_header_action",
      documented = false,
      parameters = {
        @Param(name = "actions", positional = false, named = true),
        @Param(name = "umbrella_header", positional = false, named = true),
        @Param(name = "public_headers", positional = false, named = true),
        @Param(name = "additional_exported_headers", positional = false, named = true),
      })
  public void createUmbrellaHeaderAction(
      StarlarkActionFactory actions,
      Artifact umbrellaHeader,
      Sequence<?> publicHeaders,
      Sequence<?> additionalExportedHeaders)
      throws EvalException {
    actions
        .getRuleContext()
        .registerAction(
            new UmbrellaHeaderAction(
                actions.getRuleContext().getActionOwner(),
                umbrellaHeader,
                Sequence.cast(publicHeaders, Artifact.class, "public_headers"),
                Sequence.cast(
                        additionalExportedHeaders, String.class, "additional_exported_headers")
                    .stream()
                    .map(PathFragment::create)
                    .collect(toImmutableList())));
  }

  @StarlarkMethod(
      name = "create_module_map_action",
      documented = false,
      parameters = {
        @Param(name = "actions", positional = false, named = true),
        @Param(name = "feature_configuration", positional = false, named = true),
        @Param(name = "module_map", positional = false, named = true),
        @Param(name = "private_headers", positional = false, named = true),
        @Param(name = "public_headers", positional = false, named = true),
        @Param(name = "dependent_module_maps", positional = false, named = true),
        @Param(name = "additional_exported_headers", positional = false, named = true),
        @Param(name = "separate_module_headers", positional = false, named = true),
        @Param(name = "compiled_module", positional = false, named = true),
        @Param(name = "module_map_home_is_cwd", positional = false, named = true),
        @Param(name = "generate_submodules", positional = false, named = true),
        @Param(name = "without_extern_dependencies", positional = false, named = true),
      })
  public void createModuleMapAction(
      StarlarkActionFactory actions,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      CppModuleMap moduleMap,
      Sequence<?> privateHeaders,
      Sequence<?> publicHeaders,
      Sequence<?> dependentModuleMaps,
      Sequence<?> additionalExportedHeaders,
      Sequence<?> separateModuleHeaders,
      Boolean compiledModule,
      Boolean moduleMapHomeIsCwd,
      Boolean generateSubmodules,
      Boolean withoutExternDependencies)
      throws EvalException {
    RuleContext ruleContext = actions.getRuleContext();
    ruleContext.registerAction(
        new CppModuleMapAction(
            ruleContext.getActionOwner(),
            moduleMap,
            Sequence.cast(privateHeaders, Artifact.class, "private_headers"),
            Sequence.cast(publicHeaders, Artifact.class, "public_headers"),
            Sequence.cast(dependentModuleMaps, CppModuleMap.class, "dependent_module_maps"),
            Sequence.cast(additionalExportedHeaders, String.class, "additional_exported_headers")
                .stream()
                .map(PathFragment::create)
                .collect(toImmutableList()),
            Sequence.cast(separateModuleHeaders, Artifact.class, "separate_module_headers"),
            compiledModule,
            moduleMapHomeIsCwd,
            generateSubmodules,
            withoutExternDependencies));
  }

  private static final StarlarkProvider buildSettingInfo =
      StarlarkProvider.builder(Location.BUILTIN)
          .setExported(
              new StarlarkProvider.Key(
                  Label.parseCanonicalUnchecked(
                      "//third_party/bazel_skylib/rules:common_settings.bzl"),
                  "BuildSettingInfo"))
          .build();

  @StarlarkMethod(name = "BuildSettingInfo", documented = false, structField = true)
  public StarlarkProvider buildSettingInfo() throws EvalException {
    return buildSettingInfo;
  }

  @StarlarkMethod(
      name = "escape_label",
      documented = false,
      parameters = {
        @Param(name = "label", positional = false, named = true),
      })
  public String escapeLabel(Label label) {
    return Actions.escapeLabel(label);
  }

  @StarlarkMethod(
      name = "licenses",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      },
      allowReturnNones = true)
  @Nullable
  public LicensesProvider getLicenses(StarlarkRuleContext starlarkRuleContext) {
    RuleContext ruleContext = starlarkRuleContext.getRuleContext();
    final License outputLicense =
        ruleContext.getRule().getToolOutputLicense(ruleContext.attributes());
    if (outputLicense != null && !outputLicense.equals(License.NO_LICENSE)) {
      final NestedSet<TargetLicense> license =
          NestedSetBuilder.create(
              Order.STABLE_ORDER, new TargetLicense(ruleContext.getLabel(), outputLicense));
      return new LicensesProviderImpl(
          license, new TargetLicense(ruleContext.getLabel(), outputLicense));
    } else {
      return null;
    }
  }

  @StarlarkMethod(
      name = "get_artifact_name_for_category",
      documented = false,
      parameters = {
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "category", positional = false, named = true),
        @Param(name = "output_name", positional = false, named = true),
      })
  public String getArtifactNameForCategory(Info ccToolchainInfo, String category, String outputName)
      throws RuleErrorException, EvalException {
    CcToolchainProvider ccToolchain = CcToolchainProvider.PROVIDER.wrap(ccToolchainInfo);
    return ccToolchain
        .getFeatures()
        .getArtifactNameForCategory(ArtifactCategory.valueOf(category), outputName);
  }
}
