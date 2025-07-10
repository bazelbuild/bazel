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
import static com.google.devtools.build.lib.rules.cpp.CppHelper.asDict;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.analysis.starlark.Args;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory.StarlarkActionContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.MapVariables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder.LinkActionConstruction;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.NativeComputedDefaultApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Location;

/** Utility methods for rules in Starlark Builtins */
@StarlarkBuiltin(name = "cc_internal", category = DocCategory.BUILTIN, documented = false)
public class CcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "cc_internal";

  /** Wraps a dictionary of build variables into CcToolchainVariables. */
  @StarlarkMethod(
      name = "cc_toolchain_variables",
      documented = false,
      parameters = {
        @Param(name = "vars", positional = false, named = true),
      })
  public CcToolchainVariables getCcToolchainVariables(Dict<?, ?> buildVariables)
      throws EvalException {
    return new MapVariables(null, Dict.cast(buildVariables, String.class, Object.class, "vars"));
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
      name = "dynamic_library_symlink",
      documented = false,
      parameters = {
        @Param(name = "actions"),
        @Param(name = "library"),
        @Param(name = "solib_directory"),
        @Param(name = "preserve_name"),
        @Param(name = "prefix_consumer"),
      })
  public Artifact dynamicLibrarySymlinkAction(
      StarlarkActionFactory actions,
      Artifact library,
      String solibDirectory,
      boolean preserveName,
      boolean prefixConsumer) {
    return SolibSymlinkAction.getDynamicLibrarySymlink(
        actions.getRuleContext(), solibDirectory, library, preserveName, prefixConsumer);
  }

  @StarlarkMethod(
      name = "dynamic_library_symlink2",
      documented = false,
      parameters = {
        @Param(name = "actions"),
        @Param(name = "library"),
        @Param(name = "solib_directory"),
        @Param(name = "path"),
      })
  public Artifact dynamicLibrarySymlinkAction2(
      StarlarkActionFactory actions, Artifact library, String solibDirectory, String path) {
    return SolibSymlinkAction.getDynamicLibrarySymlink(
        actions.getRuleContext(), solibDirectory, library, PathFragment.create(path));
  }

  @StarlarkMethod(
      name = "dynamic_library_soname",
      documented = false,
      parameters = {
        @Param(name = "actions"),
        @Param(name = "path"),
        @Param(name = "preserve_name"),
      })
  public String dynamicLibrarySoname(
      WrappedStarlarkActionFactory actions, String path, boolean preserveName) {

    return SolibSymlinkAction.getDynamicLibrarySoname(
        PathFragment.create(path),
        preserveName,
        actions.construction.getContext().getConfiguration().getMnemonic());
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
        .getPackageDeclarations()
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
        .getPackageDeclarations()
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
        .getPackageDeclarations()
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
        .getPackageDeclarations()
        .getPackageArgs()
        .getDefaultHdrsCheck();
  }

  @StarlarkMethod(name = "launcher_provider", documented = false, structField = true)
  public ProviderApi getCcLauncherInfoProvider() throws EvalException {
    return CcLauncherInfo.PROVIDER;
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

  @SerializationConstant @VisibleForSerialization
  static final StarlarkProvider starlarkCcTestRunnerInfo =
      StarlarkProvider.builder(Location.BUILTIN)
          .buildExported(
              new StarlarkProvider.Key(
                  keyForBuild(Label.parseCanonicalUnchecked("//tools/cpp/cc_test:toolchain.bzl")),
                  "CcTestRunnerInfo"));

  @StarlarkMethod(name = "CcTestRunnerInfo", documented = false, structField = true)
  public StarlarkProvider ccTestRunnerInfo() throws EvalException {
    return starlarkCcTestRunnerInfo;
  }

  @StarlarkMethod(
      name = "create_cpp_source",
      doc = "Creates a CppSource instance.",
      parameters = {
        @Param(
            name = "source",
            positional = false,
            named = true,
            doc = "The source file.",
            allowedTypes = {@ParamType(type = Artifact.class)}),
        @Param(
            name = "label",
            positional = false,
            named = true,
            doc = "The label of the source file.",
            allowedTypes = {@ParamType(type = Label.class)}),
        @Param(
            name = "type",
            positional = false,
            named = true,
            doc = "The type of the source file.",
            allowedTypes = {@ParamType(type = String.class)})
      })
  public CppSource createCppSource(Artifact source, Label label, String type) {
    return CppSource.create(source, label, CppSource.Type.valueOf(type));
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

  @SerializationConstant @VisibleForSerialization
  static final StarlarkProvider buildSettingInfo =
      StarlarkProvider.builder(Location.BUILTIN)
          .buildExported(
              new StarlarkProvider.Key(
                  keyForBuild(
                      Label.parseCanonicalUnchecked(
                          "//third_party/bazel_skylib/rules:common_settings.bzl")),
                  "BuildSettingInfo"));

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
  public StarlarkList<String> getLicenses(StarlarkRuleContext starlarkRuleContext) {
    return null;
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

  @StarlarkMethod(
      name = "get_artifact_name_extension_for_category",
      documented = false,
      parameters = {
        @Param(name = "cc_toolchain", named = true),
        @Param(name = "category", named = true),
      })
  public String getArtifactNameExtensionForCategory(Info ccToolchainInfo, String category)
      throws RuleErrorException, EvalException {
    CcToolchainProvider ccToolchain = CcToolchainProvider.PROVIDER.wrap(ccToolchainInfo);
    return ccToolchain
        .getFeatures()
        .getArtifactNameExtensionForCategory(ArtifactCategory.valueOf(category));
  }

  @StarlarkMethod(
      name = "absolute_symlink",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "output", positional = false, named = true),
        @Param(name = "target_path", positional = false, named = true),
        @Param(name = "progress_message", positional = false, named = true),
      })
  // TODO(b/333997009): remove command line flags that specify FDO with absolute path
  public void absoluteSymlink(
      StarlarkActionContext ctx, Artifact output, String targetPath, String progressMessage) {
    SymlinkAction action =
        SymlinkAction.toAbsolutePath(
            ctx.getRuleContext().getActionOwner(),
            PathFragment.create(targetPath),
            output,
            progressMessage);
    ctx.getRuleContext().registerAction(action);
  }

  private static final Interner<Object> interner = BlazeInterners.newWeakInterner();

  @StarlarkMethod(
      name = "intern_seq",
      documented = false,
      parameters = {@Param(name = "seq")})
  public Sequence<?> internList(Sequence<?> seq) {
    return Tuple.copyOf(Iterables.transform(seq, interner::intern));
  }

  @StarlarkMethod(
      name = "get_link_args",
      documented = false,
      parameters = {
        @Param(name = "action_name", positional = false, named = true),
        @Param(name = "feature_configuration", positional = false, named = true),
        @Param(name = "build_variables", positional = false, named = true),
        @Param(
            name = "parameter_file_type",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)}),
      })
  public Args getArgs(
      String actionName,
      FeatureConfigurationForStarlark featureConfiguration,
      CcToolchainVariables buildVariables,
      Object paramFileType)
      throws EvalException {
    LinkCommandLine.Builder linkCommandLineBuilder =
        new LinkCommandLine.Builder()
            .setActionName(actionName)
            .setBuildVariables(buildVariables)
            .setFeatureConfiguration(featureConfiguration.getFeatureConfiguration());
    if (paramFileType instanceof String) {
      linkCommandLineBuilder
          .setParameterFileType(ParameterFileType.valueOf((String) paramFileType))
          .setSplitCommandLine(true);
    }
    LinkCommandLine linkCommandLine = linkCommandLineBuilder.build();
    return Args.forRegisteredAction(
        new CommandLineAndParamFileInfo(linkCommandLine, linkCommandLine.getParamFileInfo()),
        ImmutableSet.of());
  }

  @StarlarkMethod(
      name = "setup_fdo_build_variables",
      documented = false,
      parameters = {
        @Param(name = "cc_toolchain", documented = false, positional = false, named = true),
        @Param(
            name = "fdo_context",
            documented = false,
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = StructImpl.class),
              @ParamType(type = NoneType.class)
            }),
        @Param(name = "auxiliary_fdo_inputs", documented = false, positional = false, named = true),
        @Param(
            name = "feature_configuration",
            documented = false,
            positional = false,
            named = true),
        @Param(
            name = "fdo_instrument",
            documented = false,
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)}),
        @Param(
            name = "cs_fdo_instrument",
            documented = false,
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)}),
      })
  public Dict<String, String> setupFdoBuildVariables(
      StarlarkInfo ccToolchain,
      Object fdoContextObj,
      Depset auxiliaryFdoInputs,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      Object fdoInstrument,
      Object csFdoInstrument)
      throws EvalException, TypeException {
    FdoContext fdoContext =
        fdoContextObj == Starlark.NONE ? null : new FdoContext((StructImpl) fdoContextObj);
    return Dict.immutableCopyOf(
        CcStaticCompilationHelper.setupFdoBuildVariables(
            CcToolchainProvider.create(ccToolchain),
            fdoContext,
            auxiliaryFdoInputs.getSet(Artifact.class),
            featureConfigurationForStarlark.getFeatureConfiguration(),
            CppHelper.stringFromNoneable(fdoInstrument, null),
            CppHelper.stringFromNoneable(csFdoInstrument, null)));
  }

  @StarlarkMethod(
      name = "get_auxiliary_fdo_inputs",
      documented = false,
      parameters = {
        @Param(name = "cc_toolchain", documented = false, positional = false, named = true),
        @Param(name = "fdo_context", documented = false, positional = false, named = true),
        @Param(
            name = "feature_configuration",
            documented = false,
            positional = false,
            named = true),
      })
  public Depset getAuxiliaryFdoInputs(
      StarlarkInfo ccToolchain,
      StructImpl fdoContextStruct,
      FeatureConfigurationForStarlark featureConfigurationForStarlark)
      throws EvalException {
    return Depset.of(
        Artifact.class,
        CcStaticCompilationHelper.getAuxiliaryFdoInputs(
            CcToolchainProvider.create(ccToolchain),
            new FdoContext(fdoContextStruct),
            featureConfigurationForStarlark.getFeatureConfiguration()));
  }

  @StarlarkMethod(name = "empty_compilation_outputs", documented = false)
  public CcCompilationOutputs getEmpty() {
    return CcCompilationOutputs.EMPTY;
  }

  static class WrappedStarlarkActionFactory extends StarlarkActionFactory {
    final LinkActionConstruction construction;

    public WrappedStarlarkActionFactory(
        StarlarkActionFactory parent, LinkActionConstruction construction) {
      super(parent);
      this.construction = construction;
    }

    @Override
    public FileApi createShareableArtifact(
        String path, Object artifactRoot, StarlarkThread thread) {
      return construction.create(PathFragment.create(path));
    }
  }

  @StarlarkMethod(
      name = "wrap_link_actions",
      documented = false,
      parameters = {
        @Param(name = "actions"),
        @Param(name = "build_configuration", defaultValue = "None"),
        @Param(name = "sharable_artifacts", defaultValue = "False")
      })
  public WrappedStarlarkActionFactory wrapLinkActions(
      StarlarkActionFactory actions, Object config, boolean shareableArtifacts) {
    LinkActionConstruction construction =
        CppLinkActionBuilder.newActionConstruction(
            actions.getRuleContext(),
            config instanceof BuildConfigurationValue
                ? (BuildConfigurationValue) config
                : actions.getRuleContext().getConfiguration(),
            shareableArtifacts);
    return new WrappedStarlarkActionFactory(actions, construction);
  }

  @StarlarkMethod(
      name = "actions2ctx_cheat",
      documented = false,
      parameters = {
        @Param(name = "actions"),
      })
  public StarlarkRuleContext getStarlarkRuleContext(StarlarkActionFactory actions) {
    return actions.getRuleContext().getStarlarkRuleContext();
  }

  @StarlarkMethod(
      name = "rule_kind_cheat",
      documented = false,
      parameters = {
        @Param(name = "actions"),
      })
  public String getTargetKind(StarlarkActionFactory actions) {
    return actions
        .getRuleContext()
        .getStarlarkRuleContext()
        .getRuleContext()
        .getRule()
        .getTargetKind();
  }

  @StarlarkMethod(
      name = "collect_per_file_lto_backend_opts",
      documented = false,
      parameters = {@Param(name = "cpp_config"), @Param(name = "obj")})
  public ImmutableList<String> collectPerFileLtoBackendOpts(
      CppConfiguration cppConfiguration, Artifact objectFile) {
    return cppConfiguration.getPerFileLtoBackendOpts().stream()
        .filter(perLabelOptions -> perLabelOptions.isIncluded(objectFile))
        .map(PerLabelOptions::getOptions)
        .flatMap(options -> options.stream())
        .collect(toImmutableList());
  }

  /**
   * Returns a {@code CcCompilationContext} that is based on a given {@code CcCompilationContext},
   * with {@code extraHeaderTokens} added to the header tokens.
   */
  @StarlarkMethod(
      name = "create_cc_compilation_context_with_extra_header_tokens",
      doc =
          "Creates a <code>CompilationContext</code> based on an existing one, with extra header"
              + " tokens.",
      parameters = {
        @Param(
            name = "cc_compilation_context",
            doc = "The base <code>CompilationContext</code>.",
            positional = false,
            named = true),
        @Param(
            name = "extra_header_tokens",
            doc = "The extra header tokens to add.",
            positional = false,
            named = true),
      })
  public CcCompilationContext createCcCompilationContextWithExtraHeaderTokens(
      CcCompilationContext ccCompilationContext, Sequence<?> extraHeaderTokens)
      throws EvalException {
    return CcCompilationContext.createWithExtraHeaderTokens(
        ccCompilationContext,
        Sequence.cast(extraHeaderTokens, Artifact.class, "extra_header_tokens").getImmutableList());
  }

  @StarlarkMethod(
      name = "create_copts_filter",
      doc = "Creates a copts filter from a regex.",
      documented = false,
      parameters = {
        @Param(
            name = "copts_filter",
            doc =
                "The regex to use for the copts filter. If not given, a filter that always passes"
                    + " is returned.",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            positional = true,
            named = true,
            defaultValue = "unbound"),
      })
  public StarlarkValue createCoptsFilterForStarlark(Object coptsFilterObject) throws EvalException {
    return createCoptsFilter(coptsFilterObject);
  }

  private CoptsFilter createCoptsFilter(Object coptsFilterObject) throws EvalException {
    String coptsFilterRegex =
        (Starlark.isNullOrNone(coptsFilterObject) || coptsFilterObject == Starlark.UNBOUND)
            ? null
            : (String) coptsFilterObject;
    CoptsFilter coptsFilter = null;
    if (Strings.isNullOrEmpty(coptsFilterRegex)) {
      coptsFilter = CoptsFilter.alwaysPasses();
    } else {
      try {
        coptsFilter = CoptsFilter.fromRegex(Pattern.compile(coptsFilterRegex));
      } catch (PatternSyntaxException e) {
        throw Starlark.errorf(
            "invalid regular expression '%s': %s", coptsFilterRegex, e.getMessage());
      }
    }
    return coptsFilter;
  }

  // TODO(b/396122076): Test whether this can be replaced with artifact.is_directory().
  @StarlarkMethod(
      name = "is_tree_artifact",
      documented = false,
      parameters = {
        @Param(
            name = "artifact",
            allowedTypes = {@ParamType(type = Artifact.class)})
      })
  public boolean isTreeArtifact(Artifact artifact) {
    return artifact.isTreeArtifact();
  }

  @StarlarkMethod(
      name = "create_module_action",
      documented = false,
      parameters = {
        @Param(name = "action_construction_context", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true), // StarlarkInfo
        @Param(name = "configuration", positional = false, named = true), // BuildConfigurationValue
        @Param(name = "conlyopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "copts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "copts_filter", positional = false, named = true, defaultValue = "None"),
        @Param(name = "cpp_configuration", positional = false, named = true),
        @Param(name = "cxxopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "fdo_context", positional = false, named = true, defaultValue = "None"),
        @Param(
            name = "auxiliary_fdo_inputs",
            positional = false,
            named = true,
            defaultValue = "None"),
        @Param(
            name = "feature_configuration",
            positional = false,
            named = true), // FeatureConfigurationForStarlark
        @Param(name = "generate_no_pic_action", positional = false, named = true),
        @Param(name = "generate_pic_action", positional = false, named = true),
        @Param(name = "label", positional = false, named = true),
        @Param(name = "common_toolchain_variables", positional = false, named = true),
        @Param(name = "fdo_build_variables", positional = false, named = true, defaultValue = "{}"),
        @Param(name = "cpp_semantics", positional = false, named = true), // CppSemantics
        @Param(name = "outputs", positional = false, named = true), // CcCompilationOutputs.Builder
        @Param(name = "cpp_module_map", positional = false, named = true),
        @Param(name = "cpp_compile_action_builder", positional = false, named = true),
      })
  public StarlarkList<Artifact> createModuleActionForStarlark(
      StarlarkRuleContext starlarkRuleContext,
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      Sequence<?> conlyopts, // String
      Sequence<?> copts, // String
      CoptsFilter coptsFilter,
      CppConfiguration cppConfiguration,
      Sequence<?> cxxopts, // String
      StructImpl fdoContext,
      Depset auxiliaryFdoInputs,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      boolean generateNoPicAction,
      boolean generatePicAction,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      Dict<?, ?> fdoBuildVariables, // String, String
      CppSemantics cppSemantics,
      CcCompilationOutputs.Builder outputsBuilder,
      CppModuleMap cppModuleMap,
      CppCompileActionBuilder builder)
      throws EvalException, RuleErrorException, InterruptedException {
    ImmutableList<Artifact> moduleArtifacts =
        CcStaticCompilationHelper.createModuleAction(
            starlarkRuleContext.getRuleContext(),
            ccCompilationContext,
            CcToolchainProvider.create(ccToolchain),
            configuration,
            Sequence.cast(conlyopts, String.class, "conlyopts").getImmutableList(),
            Sequence.cast(copts, String.class, "copts").getImmutableList(),
            coptsFilter,
            cppConfiguration,
            Sequence.cast(cxxopts, String.class, "cxxopts").getImmutableList(),
            TargetUtils.getExecutionInfo(
                starlarkRuleContext.getRuleContext().getRule(),
                starlarkRuleContext.getRuleContext().isAllowTagsPropagation()),
            new FdoContext(fdoContext),
            Depset.cast(auxiliaryFdoInputs, Artifact.class, "auxiliary_fdo_inputs"),
            featureConfigurationForStarlark.getFeatureConfiguration(),
            generateNoPicAction,
            generatePicAction,
            label,
            commonToolchainVariables,
            ImmutableMap.copyOf(
                Dict.cast(fdoBuildVariables, String.class, String.class, "fdo_build_variables")),
            starlarkRuleContext.getRuleContext().getRuleErrorConsumer(),
            cppSemantics,
            outputsBuilder,
            cppModuleMap,
            builder);
    return StarlarkList.immutableCopyOf(moduleArtifacts);
  }

  @StarlarkMethod(
      name = "create_module_codegen_action",
      documented = false,
      parameters = {
        @Param(name = "action_construction_context", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "conlyopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "copts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "copts_filter", positional = false, named = true, defaultValue = "None"),
        @Param(name = "cpp_configuration", positional = false, named = true),
        @Param(name = "cxxopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "fdo_context", positional = false, named = true, defaultValue = "None"),
        @Param(
            name = "auxiliary_fdo_inputs",
            positional = false,
            named = true,
            defaultValue = "None"),
        @Param(
            name = "feature_configuration",
            positional = false,
            named = true), // FeatureConfigurationForStarlark
        @Param(name = "is_code_coverage_enabled", positional = false, named = true),
        @Param(name = "label", positional = false, named = true),
        @Param(name = "common_toolchain_variables", positional = false, named = true),
        @Param(name = "fdo_build_variables", positional = false, named = true, defaultValue = "{}"),
        @Param(name = "cpp_semantics", positional = false, named = true), // CppSemantics
        @Param(name = "outputs", positional = false, named = true), // CcCompilationOutputs.Builder
        @Param(name = "source_label", positional = false, named = true),
        @Param(name = "module", positional = false, named = true),
        @Param(name = "cpp_compile_action_builder", positional = false, named = true),
      })
  public void createModuleCodegenActionForStarlark(
      StarlarkRuleContext starlarkRuleContext,
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      Sequence<?> conlyopts,
      Sequence<?> copts,
      CoptsFilter coptsFilter,
      CppConfiguration cppConfiguration,
      Sequence<?> cxxopts,
      StructImpl fdoContext,
      Depset auxiliaryFdoInputs,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      boolean isCodeCoverageEnabled,
      Label label,
      CcToolchainVariables commonToolchainVariables,
      Dict<?, ?> fdoBuildVariables,
      CppSemantics cppSemantics,
      CcCompilationOutputs.Builder outputs,
      Label sourceLabel,
      Artifact module,
      CppCompileActionBuilder builder)
      throws EvalException, RuleErrorException, InterruptedException {
    CcStaticCompilationHelper.createModuleCodegenAction(
        starlarkRuleContext.getRuleContext(),
        ccCompilationContext,
        CcToolchainProvider.create(ccToolchain),
        configuration,
        Sequence.cast(conlyopts, String.class, "conlyopts").getImmutableList(),
        Sequence.cast(copts, String.class, "copts").getImmutableList(),
        coptsFilter,
        cppConfiguration,
        Sequence.cast(cxxopts, String.class, "cxxopts").getImmutableList(),
        TargetUtils.getExecutionInfo(
            starlarkRuleContext.getRuleContext().getRule(),
            starlarkRuleContext.getRuleContext().isAllowTagsPropagation()),
        new FdoContext(fdoContext),
        Depset.cast(auxiliaryFdoInputs, Artifact.class, "auxiliary_fdo_inputs"),
        featureConfigurationForStarlark.getFeatureConfiguration(),
        isCodeCoverageEnabled,
        label,
        commonToolchainVariables,
        ImmutableMap.copyOf(
            Dict.cast(fdoBuildVariables, String.class, String.class, "fdo_build_variables")),
        starlarkRuleContext.getRuleContext().getRuleErrorConsumer(),
        cppSemantics,
        outputs,
        sourceLabel,
        module,
        builder);
  }

  @StarlarkMethod(
      name = "create_compile_source_action_from_builder",
      documented = false,
      parameters = {
        @Param(name = "action_construction_context", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "conlyopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "copts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "cpp_configuration", positional = false, named = true),
        @Param(name = "cxxopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "fdo_context", positional = false, named = true, defaultValue = "None"),
        @Param(
            name = "auxiliary_fdo_inputs",
            positional = false,
            named = true,
            defaultValue = "None"),
        @Param(
            name = "feature_configuration",
            positional = false,
            named = true), // FeatureConfigurationForStarlark
        @Param(name = "generate_no_pic_action", positional = false, named = true),
        @Param(name = "generate_pic_action", positional = false, named = true),
        @Param(name = "label", positional = false, named = true),
        @Param(name = "common_compile_build_variables", positional = false, named = true),
        @Param(name = "fdo_build_variables", positional = false, named = true, defaultValue = "{}"),
        @Param(name = "cpp_semantics", positional = false, named = true), // CppSemantics
        @Param(name = "source_label", positional = false, named = true),
        @Param(name = "output_name", positional = false, named = true),
        @Param(name = "outputs", positional = false, named = true),
        @Param(name = "source_artifact", positional = false, named = true),
        @Param(name = "cpp_compile_action_builder", positional = false, named = true),
        @Param(name = "output_category", positional = false, named = true),
        @Param(
            name = "cpp_module_map",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = CppModuleMap.class),
              @ParamType(type = NoneType.class)
            },
            defaultValue = "None"),
        @Param(name = "add_object", positional = false, named = true),
        @Param(name = "enable_coverage", positional = false, named = true),
        @Param(name = "generate_dwo", positional = false, named = true),
        @Param(name = "bitcode_output", positional = false, named = true)
      })
  public Sequence<?> createCompileSourceActionFromBuilder(
      StarlarkRuleContext starlarkRuleContext,
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      Sequence<?> conlyopts,
      Sequence<?> copts,
      CppConfiguration cppConfiguration,
      Sequence<?> cxxopts,
      StructImpl fdoContext,
      Depset auxiliaryFdoInputs,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      boolean generateNoPicAction,
      boolean generatePicAction,
      Label label,
      CcToolchainVariables commonCompileBuildVariables,
      Dict<?, ?> fdoBuildVariables,
      CppSemantics semantics,
      Label sourceLabel,
      String outputName,
      CcCompilationOutputs.Builder outputs,
      Artifact sourceArtifact,
      CppCompileActionBuilder builder,
      String outputCategoryString,
      Object cppModuleMapObject,
      boolean addObject,
      boolean enableCoverage,
      boolean generateDwo,
      boolean bitcodeOutput)
      throws RuleErrorException, EvalException, InterruptedException {
    ArtifactCategory outputCategory;
    try {
      outputCategory = ArtifactCategory.valueOf(outputCategoryString);
    } catch (IllegalArgumentException e) {
      throw new EvalException(
          String.format("Invalid output category: %s", outputCategoryString), e);
    }
    CppModuleMap cppModuleMap = null;
    if (cppModuleMapObject instanceof CppModuleMap cppModuleMapChecked) {
      cppModuleMap = cppModuleMapChecked;
    }

    return Tuple.copyOf(
        CcStaticCompilationHelper.createCompileSourceActionFromBuilder(
            starlarkRuleContext.getRuleContext(),
            ccCompilationContext,
            CcToolchainProvider.create(ccToolchain),
            configuration,
            Sequence.cast(conlyopts, String.class, "conlyopts").getImmutableList(),
            Sequence.cast(copts, String.class, "copts").getImmutableList(),
            cppConfiguration,
            Sequence.cast(cxxopts, String.class, "cxxopts").getImmutableList(),
            new FdoContext(fdoContext),
            Depset.cast(auxiliaryFdoInputs, Artifact.class, "auxiliary_fdo_inputs"),
            featureConfigurationForStarlark.getFeatureConfiguration(),
            generateNoPicAction,
            generatePicAction,
            label,
            commonCompileBuildVariables,
            ImmutableMap.copyOf(
                Dict.cast(fdoBuildVariables, String.class, String.class, "fdo_build_variables")),
            starlarkRuleContext.getRuleContext().getRuleErrorConsumer(),
            semantics,
            sourceLabel,
            outputName,
            outputs,
            sourceArtifact,
            builder,
            outputCategory,
            cppModuleMap,
            addObject,
            enableCoverage,
            generateDwo,
            bitcodeOutput));
  }

  @StarlarkMethod(
      name = "create_compile_action_template",
      documented = false,
      parameters = {
        @Param(name = "action_construction_context", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "conlyopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "copts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "cpp_configuration", positional = false, named = true),
        @Param(name = "cxxopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "fdo_context", positional = false, named = true, defaultValue = "None"),
        @Param(
            name = "auxiliary_fdo_inputs",
            positional = false,
            named = true,
            defaultValue = "None"),
        @Param(
            name = "feature_configuration",
            positional = false,
            named = true), // FeatureConfigurationForStarlark
        @Param(name = "label", positional = false, named = true),
        @Param(name = "common_compile_build_variables", positional = false, named = true),
        @Param(name = "fdo_build_variables", positional = false, named = true),
        @Param(name = "cpp_semantics", positional = false, named = true),
        @Param(name = "source", positional = false, named = true),
        @Param(name = "output_name", positional = false, named = true),
        @Param(name = "cpp_compile_action_builder", positional = false, named = true),
        @Param(name = "outputs", positional = false, named = true),
        @Param(name = "output_categories", positional = false, named = true),
        @Param(name = "use_pic", positional = false, named = true),
        @Param(name = "bitcode_output", positional = false, named = true)
      })
  public Artifact createCompileActionTemplate(
      StarlarkRuleContext starlarkRuleContext,
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      Sequence<?> conlyopts,
      Sequence<?> copts,
      CppConfiguration cppConfiguration,
      Sequence<?> cxxopts,
      StructImpl fdoContext,
      Depset auxiliaryFdoInputs,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      Label label,
      CcToolchainVariables commonCompileBuildVariables,
      Dict<?, ?> fdoBuildVariables,
      CppSemantics semantics,
      CppSource source,
      String outputName,
      CppCompileActionBuilder builder,
      CcCompilationOutputs.Builder outputs,
      Sequence<?> outputCategoriesUnchecked,
      boolean usePic,
      boolean bitcodeOutput)
      throws RuleErrorException, EvalException {
    ImmutableList.Builder<ArtifactCategory> outputCategories = ImmutableList.builder();
    for (Object outputCategoryObject : outputCategoriesUnchecked) {
      if (outputCategoryObject instanceof String outputCategoryString) {
        try {
          outputCategories.add(ArtifactCategory.valueOf(outputCategoryString));
        } catch (IllegalArgumentException e) {
          EvalException evalException =
              new EvalException(String.format("Invalid output category: %s", outputCategoryObject));
          evalException.initCause(e);
          throw evalException;
        }
      } else {
        throw new EvalException(
            String.format(
                "Output category has invalid type. Expected string, got: %s",
                outputCategoryObject));
      }
    }
    return CcStaticCompilationHelper.createCompileActionTemplate(
        starlarkRuleContext.getRuleContext(),
        ccCompilationContext,
        CcToolchainProvider.create(ccToolchain),
        configuration,
        Sequence.cast(conlyopts, String.class, "conlyopts").getImmutableList(),
        Sequence.cast(copts, String.class, "copts").getImmutableList(),
        cppConfiguration,
        Sequence.cast(cxxopts, String.class, "cxxopts").getImmutableList(),
        new FdoContext(fdoContext),
        Depset.cast(auxiliaryFdoInputs, Artifact.class, "auxiliary_fdo_inputs"),
        featureConfigurationForStarlark.getFeatureConfiguration(),
        label,
        commonCompileBuildVariables,
        ImmutableMap.copyOf(
            Dict.cast(fdoBuildVariables, String.class, String.class, "fdo_build_variables")),
        starlarkRuleContext.getRuleContext().getRuleErrorConsumer(),
        semantics,
        source,
        outputName,
        builder,
        outputs,
        outputCategories.build(),
        usePic,
        bitcodeOutput);
  }

  @StarlarkMethod(
      name = "create_parse_header_action",
      documented = false,
      parameters = {
        @Param(name = "action_construction_context", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "conlyopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "copts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "cpp_configuration", positional = false, named = true),
        @Param(name = "cxxopts", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "fdo_context", positional = false, named = true, defaultValue = "None"),
        @Param(
            name = "auxiliary_fdo_inputs",
            positional = false,
            named = true,
            defaultValue = "None"),
        @Param(
            name = "feature_configuration",
            positional = false,
            named = true), // FeatureConfigurationForStarlark
        @Param(name = "use_pic", positional = false, named = true),
        @Param(name = "label", positional = false, named = true),
        @Param(name = "common_compile_build_variables", positional = false, named = true),
        @Param(name = "fdo_build_variables", positional = false, named = true),
        @Param(name = "cpp_semantics", positional = false, named = true),
        @Param(name = "source_label", positional = false, named = true),
        @Param(name = "output_name", positional = false, named = true),
        @Param(name = "outputs", positional = false, named = true),
        @Param(name = "cpp_compile_action_builder", positional = false, named = true)
      })
  public void createParseHeaderAction(
      StarlarkRuleContext starlarkRuleContext,
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      Sequence<?> conlyopts,
      Sequence<?> copts,
      CppConfiguration cppConfiguration,
      Sequence<?> cxxopts,
      StructImpl fdoContext,
      Depset auxiliaryFdoInputs,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      boolean generatePicAction,
      Label label,
      CcToolchainVariables commonCompileBuildVariables,
      Dict<?, ?> fdoBuildVariables,
      CppSemantics semantics,
      Label sourceLabel,
      String outputName,
      CcCompilationOutputs.Builder outputs,
      CppCompileActionBuilder builder)
      throws RuleErrorException, EvalException {
    CcStaticCompilationHelper.createParseHeaderAction(
        starlarkRuleContext.getRuleContext(),
        ccCompilationContext,
        CcToolchainProvider.create(ccToolchain),
        configuration,
        Sequence.cast(conlyopts, String.class, "conlyopts").getImmutableList(),
        Sequence.cast(copts, String.class, "copts").getImmutableList(),
        cppConfiguration,
        Sequence.cast(cxxopts, String.class, "cxxopts").getImmutableList(),
        new FdoContext(fdoContext),
        Depset.cast(auxiliaryFdoInputs, Artifact.class, "auxiliary_fdo_inputs"),
        featureConfigurationForStarlark.getFeatureConfiguration(),
        generatePicAction,
        label,
        commonCompileBuildVariables,
        ImmutableMap.copyOf(
            Dict.cast(fdoBuildVariables, String.class, String.class, "fdo_build_variables")),
        starlarkRuleContext.getRuleContext().getRuleErrorConsumer(),
        semantics,
        sourceLabel,
        outputName,
        outputs,
        builder);
  }

  @StarlarkMethod(
      name = "create_cc_compilation_outputs_builder",
      documented = false,
      parameters = {})
  public CcCompilationOutputs.Builder createCcCompilationOutputsBuilder() {
    return CcCompilationOutputs.builder();
  }

  @StarlarkMethod(
      name = "compute_output_name_prefix_dir",
      documented = false,
      parameters = {
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "purpose", positional = false, named = true),
      })
  public String computeOutputNamePrefixDir(BuildConfigurationValue configuration, String purpose) {
    return Objects.requireNonNullElse(
        CcStaticCompilationHelper.computeOutputNamePrefixDir(configuration, purpose), "");
  }

  @StarlarkMethod(
      name = "setup_common_compile_build_variables",
      documented = false,
      parameters = {
        @Param(
            name = "cc_compilation_context",
            documented = false,
            positional = false,
            named = true),
        @Param(name = "cc_toolchain", documented = false, positional = false, named = true),
        @Param(name = "cpp_configuration", documented = false, positional = false, named = true),
        @Param(name = "fdo_context", documented = false, positional = false, named = true),
        @Param(
            name = "feature_configuration",
            documented = false,
            positional = false,
            named = true),
        @Param(name = "variables_extension", documented = false, positional = false, named = true),
      })
  public CcToolchainVariables setupCommonCompileBuildVariables(
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      CppConfiguration cppConfiguration,
      StructImpl fdoContextStruct,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      Object variablesExtension)
      throws RuleErrorException, EvalException {
    ImmutableList<VariablesExtension> variablesExtensionsList =
        asDict(variablesExtension).isEmpty()
            ? ImmutableList.of()
            : ImmutableList.of(new UserVariablesExtension(asDict(variablesExtension)));
    return CcStaticCompilationHelper.setupCommonCompileBuildVariables(
        ccCompilationContext,
        CcToolchainProvider.create(ccToolchain),
        cppConfiguration,
        new FdoContext(fdoContextStruct),
        featureConfigurationForStarlark.getFeatureConfiguration(),
        variablesExtensionsList);
  }

  /**
   * Returns a {@code CppCompileActionBuilder} with the common fields for a C++ compile action being
   * initialized.
   */
  @StarlarkMethod(
      name = "create_cpp_compile_action_builder",
      documented = false,
      parameters = {
        @Param(name = "action_construction_context", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "copts_filter", positional = false, named = true),
        @Param(name = "feature_configuration", positional = false, named = true),
        @Param(name = "semantics", positional = false, named = true),
        @Param(name = "source_artifact", positional = false, named = true),
      })
  public CppCompileActionBuilder createCppCompileActionBuilder(
      StarlarkRuleContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      CoptsFilter coptsFilter,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      CppSemantics semantics,
      Artifact sourceArtifact) {
    return new CppCompileActionBuilder(
            actionConstructionContext.getRuleContext(),
            CcToolchainProvider.create(ccToolchain),
            configuration,
            semantics)
        .setSourceFile(sourceArtifact)
        .setCcCompilationContext(ccCompilationContext)
        .setCoptsFilter(coptsFilter)
        .setFeatureConfiguration(featureConfigurationForStarlark.getFeatureConfiguration())
        .addExecutionInfo(
            TargetUtils.getExecutionInfo(
                actionConstructionContext.getRuleContext().getRule(),
                actionConstructionContext.getRuleContext().isAllowTagsPropagation()));
  }

  /**
   * Returns a {@code CppCompileActionBuilder} with the common fields for a C++ compile action being
   * initialized, plus the mandatoryInputs and additionalIncludeScanningRoots fields
   */
  @StarlarkMethod(
      name = "create_cpp_compile_action_builder_with_inputs",
      documented = false,
      parameters = {
        @Param(name = "action_construction_context", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "copts_filter", positional = false, named = true),
        @Param(name = "feature_configuration", positional = false, named = true),
        @Param(name = "semantics", positional = false, named = true),
        @Param(name = "source_artifact", positional = false, named = true),
        @Param(name = "additional_compilation_inputs", positional = false, named = true),
        @Param(name = "additional_include_scanning_roots", positional = false, named = true),
      })
  public CppCompileActionBuilder createCppCompileActionBuilderWithInputs(
      StarlarkRuleContext actionConstructionContext,
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      CoptsFilter coptsFilter,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      CppSemantics semantics,
      Artifact sourceArtifact,
      Sequence<?> additionalCompilationInputs,
      Sequence<?> additionalIncludeScanningRoots)
      throws EvalException {
    CppCompileActionBuilder builder =
        createCppCompileActionBuilder(
            actionConstructionContext,
            ccCompilationContext,
            ccToolchain,
            configuration,
            coptsFilter,
            featureConfigurationForStarlark,
            semantics,
            sourceArtifact);
    builder
        .addMandatoryInputs(
            Sequence.cast(
                additionalCompilationInputs, Artifact.class, "additional_compilation_inputs"))
        .addAdditionalIncludeScanningRoots(
            Sequence.cast(
                additionalIncludeScanningRoots,
                Artifact.class,
                "additional_include_scanning_roots"));
    return builder;
  }

  // TODO(b/420530680): remove after removing uses of depsets of LibraryToLink-s, LinkerInputs
  @StarlarkMethod(
      name = "freeze",
      documented = false,
      parameters = {@Param(name = "value")})
  public Object freeze(StarlarkValue value) throws InterruptedException, EvalException {
    return switch (value) {
      case Dict<?, ?> dict -> Dict.immutableCopyOf(dict);
      case Iterable<?> iterable -> StarlarkList.immutableCopyOf(iterable);
      case Object val -> val;
    };
  }

  @StarlarkMethod(
      name = "check_toplevel",
      documented = false,
      parameters = {@Param(name = "fn")})
  public void checkToplevel(StarlarkFunction fn) throws EvalException {
    if (fn.getModule().getGlobal(fn.getName()) != fn) {
      throw Starlark.errorf("Passed function must be top-level functions.");
    }
  }
}
