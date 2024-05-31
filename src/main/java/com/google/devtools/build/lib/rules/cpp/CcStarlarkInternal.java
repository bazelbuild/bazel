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
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.common.base.Ascii;
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
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory.StarlarkActionContext;
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
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.SequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariableValue;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder.LinkActionConstruction;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.NativeComputedDefaultApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.Location;

/** Utility methods for rules in Starlark Builtins */
@StarlarkBuiltin(name = "cc_internal", category = DocCategory.BUILTIN, documented = false)
public class CcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "cc_internal";

  /**
   * Wraps a dictionary of build variables into CcToolchainVariables.
   *
   * <p>TODO(b/338618120): This code helps during the transition of cc_common.link and
   * cc_common.compile code to Starlark. Once that code is in Starlark, CcToolchainVariables rewrite
   * may commence, most likely turning them into a regular Starlark dict (or a dict with parent if
   * that optimisation is still needed).
   */
  @StarlarkMethod(
      name = "cc_toolchain_variables",
      documented = false,
      parameters = {
        @Param(name = "vars", positional = false, named = true),
      })
  @SuppressWarnings("unchecked")
  public CcToolchainVariables getCcToolchainVariables(Dict<?, ?> buildVariables)
      throws TypeException {

    CcToolchainVariables.Builder ccToolchainVariables = CcToolchainVariables.builder();
    for (var entry : buildVariables.entrySet()) {
      if (entry.getValue() instanceof String) {
        ccToolchainVariables.addStringVariable((String) entry.getKey(), (String) entry.getValue());
      } else if (entry.getValue() instanceof Boolean) {
        ccToolchainVariables.addBooleanValue((String) entry.getKey(), (Boolean) entry.getValue());
      } else if (entry.getValue() instanceof Iterable<?>) {
        if (entry.getKey().equals("libraries_to_link")) {
          SequenceBuilder sb = new SequenceBuilder();
          for (var value : (Iterable<?>) entry.getValue()) {
            sb.addValue((VariableValue) value);
          }
          ccToolchainVariables.addCustomBuiltVariable((String) entry.getKey(), sb);
        } else {
          ccToolchainVariables.addStringSequenceVariable(
              (String) entry.getKey(), (Iterable<String>) entry.getValue());
        }
      } else if (entry.getValue() instanceof Depset) {
        ccToolchainVariables.addStringSequenceVariable(
            (String) entry.getKey(), ((Depset) entry.getValue()).getSet(String.class));
      }
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

  @StarlarkMethod(
      name = "for_static_library",
      documented = false,
      parameters = {@Param(name = "name"), @Param(name = "is_whole_archive", named = true)})
  public LibraryToLinkValue forStaticLibrary(String name, boolean isWholeArchive) {
    return LibraryToLinkValue.forStaticLibrary(name, isWholeArchive);
  }

  @StarlarkMethod(
      name = "for_object_file_group",
      documented = false,
      parameters = {@Param(name = "files"), @Param(name = "is_whole_archive", named = true)})
  public LibraryToLinkValue forObjectFileGroup(Sequence<?> files, boolean isWholeArchive)
      throws EvalException {
    return LibraryToLinkValue.forObjectFileGroup(
        ImmutableList.copyOf(Sequence.cast(files, Artifact.class, "files")), isWholeArchive);
  }

  @StarlarkMethod(
      name = "for_object_file",
      documented = false,
      parameters = {@Param(name = "name"), @Param(name = "is_whole_archive", named = true)})
  public LibraryToLinkValue forObjectFile(String name, boolean isWholeArchive) {
    return LibraryToLinkValue.forObjectFile(name, isWholeArchive);
  }

  @StarlarkMethod(
      name = "for_interface_library",
      documented = false,
      parameters = {@Param(name = "name")})
  public LibraryToLinkValue forInterfaceLibrary(String name) throws EvalException {
    return LibraryToLinkValue.forInterfaceLibrary(name);
  }

  @StarlarkMethod(
      name = "for_dynamic_library",
      documented = false,
      parameters = {@Param(name = "name")})
  public LibraryToLinkValue forDynamicLibrary(String name) throws EvalException {
    return LibraryToLinkValue.forDynamicLibrary(name);
  }

  @StarlarkMethod(
      name = "for_versioned_dynamic_library",
      documented = false,
      parameters = {@Param(name = "name"), @Param(name = "path")})
  public LibraryToLinkValue forVersionedDynamicLibrary(String name, String path)
      throws EvalException {
    return LibraryToLinkValue.forVersionedDynamicLibrary(name, path);
  }

  @StarlarkMethod(
      name = "simple_linker_input",
      documented = false,
      parameters = {
        @Param(name = "input"),
        @Param(name = "artifact_category", defaultValue = "'object_file'"),
        @Param(name = "disable_whole_archive", defaultValue = "False")
      })
  public LegacyLinkerInput simpleLinkerInput(
      Artifact input, String artifactCategory, boolean disableWholeArchive) {
    return LegacyLinkerInputs.simpleLinkerInput(
        input,
        ArtifactCategory.valueOf(Ascii.toUpperCase(artifactCategory)),
        /* disableWholeArchive= */ disableWholeArchive,
        input.getRootRelativePathString());
  }

  @StarlarkMethod(
      name = "linkstamp_linker_input",
      documented = false,
      parameters = {
        @Param(name = "input"),
      })
  public LegacyLinkerInput linkstampLinkerInput(Artifact input) {
    return LegacyLinkerInputs.linkstampLinkerInput(input);
  }

  @StarlarkMethod(
      name = "library_linker_input",
      documented = false,
      parameters = {
        @Param(name = "input", named = true),
        @Param(name = "artifact_category", named = true),
        @Param(name = "library_identifier", named = true),
        @Param(name = "object_files", named = true),
        @Param(name = "lto_compilation_context", named = true),
        @Param(name = "shared_non_lto_backends", defaultValue = "None", named = true),
        @Param(name = "must_keep_debug", defaultValue = "False", named = true),
        @Param(name = "disable_whole_archive", defaultValue = "False", named = true),
      })
  public LegacyLinkerInput libraryLinkerInput(
      Artifact input,
      String artifactCategory,
      String libraryIdentifier,
      Object objectFiles,
      Object ltoCompilationContext,
      Object sharedNonLtoBackends,
      boolean mustKeepDebug,
      boolean disableWholeArchive)
      throws EvalException {
    return LegacyLinkerInputs.newInputLibrary(
        input,
        ArtifactCategory.valueOf(artifactCategory),
        libraryIdentifier,
        objectFiles == Starlark.NONE
            ? null
            : Sequence.cast(objectFiles, Artifact.class, "object_files").getImmutableList(),
        ltoCompilationContext instanceof LtoCompilationContext lto ? lto : null,
        /* sharedNonLtoBackends= */ ImmutableMap.copyOf(
            Dict.noneableCast(
                sharedNonLtoBackends,
                Artifact.class,
                LtoBackendArtifacts.class,
                "shared_non_lto_backends")),
        mustKeepDebug,
        disableWholeArchive);
  }

  @StarlarkMethod(
      name = "solib_linker_input",
      documented = false,
      parameters = {
        @Param(name = "solib_symlink", named = true),
        @Param(name = "original", named = true),
        @Param(name = "library_identifier", named = true),
      })
  public LegacyLinkerInput solibLinkerInput(
      Artifact solibSymlink, Artifact original, String libraryIdentifier) throws EvalException {
    return LegacyLinkerInputs.solibLibraryInput(solibSymlink, original, libraryIdentifier);
  }

  @StarlarkMethod(name = "empty_compilation_outputs", documented = false)
  public CcCompilationOutputs getEmpty() {
    return CcCompilationOutputs.EMPTY;
  }

  private static class WrappedStarlarkActionFactory extends StarlarkActionFactory {
    private final LinkActionConstruction construction;

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
}
