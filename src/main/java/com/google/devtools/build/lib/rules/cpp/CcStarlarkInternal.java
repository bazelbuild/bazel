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
import static com.google.devtools.build.lib.analysis.constraints.ConstraintConstants.getOsFromConstraintsOrHost;
import static com.google.devtools.build.lib.rules.cpp.CcModule.nullIfNone;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.Expander;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.analysis.starlark.Args;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory.StarlarkActionContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTemplateContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CoptsFilter;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext.HeaderInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.MapVariables;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder.LinkActionConstruction;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.NativeComputedDefaultApi;
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
import net.starlark.java.eval.Module;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Tuple;

/** Utility methods for rules in Starlark Builtins */
@StarlarkBuiltin(name = "cc_internal", category = DocCategory.BUILTIN, documented = false)
public class CcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "cc_internal";

  @StarlarkMethod(
      name = "check_private_api",
      documented = false,
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "allowlist",
            documented = false,
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = Tuple.class),
            }),
        @Param(
            name = "depth",
            documented = false,
            positional = false,
            named = true,
            defaultValue = "1"),
      })
  public void checkPrivateApi(Object allowlistObject, Object depth, StarlarkThread thread)
      throws EvalException {
    // This method may be called anywhere from builtins, but not outside (because it's not exposed
    // in cc_common.bzl
    Module module =
        Module.ofInnermostEnclosingStarlarkFunction(
            thread, depth == null ? 1 : ((StarlarkInt) depth).toIntUnchecked());
    if (module == null) {
      // The module is null when the call is coming from one of the callbacks passed to execution
      // phase
      return;
    }
    BazelModuleContext bazelModuleContext = (BazelModuleContext) module.getClientData();
    ImmutableList<BuiltinRestriction.AllowlistEntry> allowlist =
        Sequence.cast(allowlistObject, Tuple.class, "allowlist").stream()
            // TODO(bazel-team): Avoid unchecked indexing and casts on values obtained from
            // Starlark, even though it is allowlisted.
            .map(p -> BuiltinRestriction.allowlistEntry((String) p.get(0), (String) p.get(1)))
            .collect(toImmutableList());
    BuiltinRestriction.failIfModuleOutsideAllowlist(bazelModuleContext, allowlist);
  }

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
      name = "combine_cc_toolchain_variables",
      documented = false,
      parameters = {
        @Param(
            name = "parent",
            allowedTypes = {@ParamType(type = CcToolchainVariables.class)})
      },
      extraPositionals =
          @Param(
              name = "variables",
              allowedTypes = {
                @ParamType(type = Sequence.class, generic1 = CcToolchainVariables.class)
              }))
  public CcToolchainVariables combineCcToolchainVariables(
      CcToolchainVariables parent, Sequence<?> variablesSequenceUnchecked) throws EvalException {
    Sequence<CcToolchainVariables> variablesSequence =
        Sequence.cast(variablesSequenceUnchecked, CcToolchainVariables.class, "variables");
    CcToolchainVariables.Builder builder = CcToolchainVariables.builder(parent);
    for (CcToolchainVariables variables : variablesSequence) {
      builder.addAllNonTransitive(variables);
    }
    return builder.build();
  }

  @StarlarkMethod(
      name = "intern_string_sequence_variable_value",
      documented = false,
      parameters = {
        @Param(name = "string_sequence"),
      })
  public Sequence<String> internStringSequenceVariableValue(Sequence<?> stringSequence)
      throws EvalException {
    return Sequence.cast(
        interner.intern(
            StarlarkList.immutableCopyOf(
                Sequence.cast(stringSequence, String.class, "string_sequence"))),
        String.class,
        "string_sequence");
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
      StarlarkInfo ccToolchainConfigInfo, String toolsDirectoryPathString)
      throws EvalException, RuleErrorException {
    return new CcToolchainFeatures(
        CcToolchainConfigInfo.PROVIDER.wrap(ccToolchainConfigInfo),
        PathFragment.create(toolsDirectoryPathString));
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
      name = "get_artifact_name_for_category",
      documented = false,
      parameters = {
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "category", positional = false, named = true),
        @Param(name = "output_name", positional = false, named = true),
      })
  public String getArtifactNameForCategory(Info ccToolchainInfo, String category, String outputName)
      throws RuleErrorException, EvalException {
    CcToolchainProvider ccToolchain = CcToolchainProvider.wrap(ccToolchainInfo);
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
    CcToolchainProvider ccToolchain = CcToolchainProvider.wrap(ccToolchainInfo);
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

    @StarlarkMethod(
        name = "declare_shareable_directory",
        parameters = {@Param(name = "path")},
        documented = false)
    public FileApi createShareableDirectory(String path) {
      return construction.createTreeArtifact(PathFragment.create(path));
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
      name = "exec_os",
      documented = false,
      parameters = {@Param(name = "ctx")})
  public String getExecOs(StarlarkRuleContext ctx) {
    return getOsFromConstraintsOrHost(ctx.getRuleContext().getExecutionPlatform()).name();
  }

  @StarlarkMethod(
      name = "rule_class",
      documented = false,
      parameters = {@Param(name = "ctx")})
  public String getRuleClass(StarlarkRuleContext ctx) {
    return ctx.getRuleContext().getRule().getRuleClass();
  }

  @StarlarkMethod(
      name = "aspect_class",
      documented = false,
      parameters = {@Param(name = "ctx")},
      allowReturnNones = true)
  @Nullable
  public String getAspectClass(StarlarkRuleContext ctx) {
    if (ctx.getAspectDescriptor() == null) {
      return null;
    }
    String aspectName = ctx.getAspectDescriptor().getAspectClass().getName();
    // Starlark aspects names are of the form //my/aspect.bzl%aspect
    if (aspectName.contains("%")) {
      aspectName = aspectName.split("%", -1)[1];
    }
    return aspectName;
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
      name = "compute_output_name_prefix_dir",
      documented = false,
      parameters = {
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "purpose", positional = false, named = true),
      })
  public String computeOutputNamePrefixDir(BuildConfigurationValue configuration, String purpose) {
    String outputNamePrefixDir = null;
    // purpose is only used by objc rules; if set it ends with either "_non_objc_arc" or
    // "_objc_arc", and it is used to override configuration.getMnemonic() to prefix the output
    // dir with "non_arc" or "arc".
    String mnemonic = configuration.getMnemonic();
    if (purpose != null) {
      mnemonic = purpose;
    }
    if (mnemonic.endsWith("_objc_arc")) {
      outputNamePrefixDir = mnemonic.endsWith("_non_objc_arc") ? "non_arc" : "arc";
    }
    return Objects.requireNonNullElse(outputNamePrefixDir, "");
  }

  @StarlarkMethod(
      name = "create_cc_compile_action",
      documented = false,
      parameters = {
        @Param(
            name = "action_construction_context",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = StarlarkRuleContext.class),
              @ParamType(type = StarlarkTemplateContext.class)
            }),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(
            name = "copts_filter",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None"),
        @Param(name = "feature_configuration", positional = false, named = true),
        @Param(name = "source", positional = false, named = true),
        @Param(
            name = "additional_compilation_inputs",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "additional_compilation_inputs_set",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Depset.class, generic1 = Artifact.class),
              @ParamType(type = NoneType.class)
            },
            defaultValue = "None"),
        @Param(
            name = "additional_include_scanning_roots",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(name = "output_file", positional = false, named = true, defaultValue = "None"),
        @Param(name = "dotd_file", positional = false, named = true, defaultValue = "None"),
        @Param(name = "diagnostics_file", positional = false, named = true, defaultValue = "None"),
        @Param(name = "gcno_file", positional = false, named = true, defaultValue = "None"),
        @Param(name = "dwo_file", positional = false, named = true, defaultValue = "None"),
        @Param(name = "lto_indexing_file", positional = false, named = true, defaultValue = "None"),
        @Param(name = "use_pic", positional = false, named = true, defaultValue = "False"),
        @Param(name = "compile_build_variables", positional = false, named = true),
        @Param(
            name = "cache_key_inputs",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Depset.class, generic1 = Artifact.class),
              @ParamType(type = NoneType.class)
            },
            defaultValue = "None"),
        @Param(
            name = "build_info_header_files",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = Artifact.class),
              @ParamType(type = NoneType.class)
            },
            defaultValue = "None"),
        @Param(
            name = "additional_prunable_headers",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Depset.class, generic1 = Artifact.class),
              @ParamType(type = NoneType.class)
            },
            defaultValue = "None"),
        @Param(
            name = "action_name",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None"),
        @Param(
            name = "should_scan_includes",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Boolean.class), @ParamType(type = NoneType.class)},
            defaultValue = "None"),
        @Param(
            name = "shareable",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Boolean.class), @ParamType(type = NoneType.class)},
            defaultValue = "None"),
        @Param(name = "module_files", positional = false, named = true, defaultValue = "None"),
        @Param(name = "modmap_file", positional = false, named = true, defaultValue = "None"),
        @Param(name = "modmap_input_file", positional = false, named = true, defaultValue = "None"),
        @Param(name = "additional_outputs", positional = false, named = true, defaultValue = "[]"),
        @Param(
            name = "needs_include_validation",
            positional = false,
            named = true,
            defaultValue = "False"),
        @Param(name = "toolchain_type", positional = false, named = true),
      })
  public void createCppCompileAction(
      Object actionConstructionContextUnchecked,
      StarlarkInfo ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      Object coptsFilterObject,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      Artifact sourceArtifact,
      Sequence<?> additionalCompilationInputs,
      Object additionalCompilationInputsSet,
      Sequence<?> additionalIncludeScanningRoots,
      Object outputFile,
      Object dotdFile,
      Object diagnosticsFile,
      Object gcnoFile,
      Object dwoFile,
      Object ltoIndexingFile,
      boolean usePic,
      CcToolchainVariables compileBuildVariables,
      Object cacheKeyInputs,
      Object buildInfoHeaderArtifacts,
      Object additionalPrunableHeaders,
      Object actionName,
      Object shouldScanIncludes,
      Object shareable,
      Object moduleFiles,
      Object modmapFile,
      Object modmapInputFile,
      Sequence<?> additionalOutputs,
      boolean needsIncludeValidation,
      String toolchainType)
      throws EvalException, TypeException {
    CcActionContext ccActionContext;
    if (actionConstructionContextUnchecked instanceof StarlarkRuleContext starlarkRuleContext) {
      ccActionContext = new CcRuleContext(starlarkRuleContext.getRuleContext(), toolchainType);
    } else if (actionConstructionContextUnchecked
        instanceof StarlarkTemplateContext starlarkTemplateContext) {
      ccActionContext = new CcTemplateContext(starlarkTemplateContext);
    } else {
      throw new EvalException(
          "action_construction_context must be either StarlarkRuleContext or"
              + " StarlarkTemplateContext");
    }
    CoptsFilter coptsFilter =
        createCoptsFilter(
            Starlark.isNullOrNone(coptsFilterObject) ? null : (String) coptsFilterObject);
    CppCompileActionBuilder builder =
        createCppCompileActionBuilder(
            ccActionContext.getActionOwner(),
            CcCompilationContext.of(ccCompilationContext),
            ccToolchain,
            configuration,
            coptsFilter,
            featureConfigurationForStarlark,
            sourceArtifact,
            additionalCompilationInputs,
            additionalIncludeScanningRoots,
            outputFile,
            dotdFile,
            diagnosticsFile,
            gcnoFile,
            dwoFile,
            ltoIndexingFile,
            usePic,
            needsIncludeValidation,
            ccActionContext.getExecutionInfo());
    if (additionalCompilationInputsSet instanceof Depset additionalCompilationInputsDepset) {
      builder.addMandatoryInputs(additionalCompilationInputsDepset.getSet(Artifact.class));
    }
    builder.setVariables(compileBuildVariables);
    if (cacheKeyInputs != Starlark.NONE) {
      builder.setCacheKeyInputs(Depset.cast(cacheKeyInputs, Artifact.class, "cache_key_inputs"));
    }
    if (buildInfoHeaderArtifacts != Starlark.NONE) {
      builder.setBuildInfoHeaderArtifacts(
          Sequence.cast(buildInfoHeaderArtifacts, Artifact.class, "builtin_header_files")
              .getImmutableList());
    }
    if (actionName instanceof String actionNameString) {
      builder.setActionName(actionNameString);
    }
    if (shouldScanIncludes instanceof Boolean bool) {
      builder.setShouldScanIncludes(bool);
    }
    if (shareable instanceof Boolean bool) {
      builder.setShareable(bool);
    }
    if (additionalPrunableHeaders instanceof Depset additionalPrunableHeadersDepset) {
      builder.setAdditionalPrunableHeaders(additionalPrunableHeadersDepset.getSet(Artifact.class));
    }
    builder.setModuleFiles(Depset.noneableCast(moduleFiles, Artifact.class, "module_files"));
    builder.setModmapFile(nullIfNone(modmapFile, Artifact.class));
    builder.setModmapInputFile(nullIfNone(modmapInputFile, Artifact.class));
    builder.setAdditionalOutputs(
        Sequence.cast(additionalOutputs, Artifact.class, "additional_outputs").getImmutableList());
    try {
      CppCompileAction compileAction = builder.buildAndVerify();
      ccActionContext.registerAction(compileAction);
    } catch (CppCompileActionBuilder.UnconfiguredActionConfigException e) {
      throw new EvalException(
          String.format(
              "Expected action_config for '%s' to be configured", builder.getActionName()),
          e);
    }
  }

  // CcActionContext encapsulates the differences between using a RuleContext (the regular case) vs.
  // a StarlarkTemplateContext (invoked inside map_directory()) when creating a CppCompileAction.
  private interface CcActionContext {
    void registerAction(CppCompileAction action);

    ActionOwner getActionOwner();

    ImmutableMap<String, String> getExecutionInfo();
  }

  private static class CcRuleContext implements CcActionContext {
    private final RuleContext ruleContext;
    private final String toolchainType;

    private CcRuleContext(RuleContext ruleContext, String toolchainType) {
      this.ruleContext = ruleContext;
      this.toolchainType = toolchainType;
    }

    @Override
    public void registerAction(CppCompileAction action) {
      ruleContext.registerAction(action);
    }

    @Override
    public ActionOwner getActionOwner() {
      ActionOwner actionOwner = null;
      if (ruleContext.useAutoExecGroups()) {
        actionOwner =
            ruleContext.getActionOwner(Label.parseCanonicalUnchecked(toolchainType).toString());
      }
      return actionOwner == null ? ruleContext.getActionOwner() : actionOwner;
    }

    @Override
    public ImmutableMap<String, String> getExecutionInfo() {
      return TargetUtils.getExecutionInfo(
          ruleContext.getRule(), ruleContext.isAllowTagsPropagation());
    }
  }

  private static class CcTemplateContext implements CcActionContext {
    private final StarlarkTemplateContext starlarkTemplateContext;

    private CcTemplateContext(StarlarkTemplateContext starlarkTemplateContext) {
      this.starlarkTemplateContext = starlarkTemplateContext;
    }

    @Override
    public void registerAction(CppCompileAction action) {
      starlarkTemplateContext.registerAction(action);
    }

    @Override
    public ActionOwner getActionOwner() {
      return starlarkTemplateContext.getActionOwner();
    }

    @Override
    public ImmutableMap<String, String> getExecutionInfo() {
      return starlarkTemplateContext.getExecutionInfo();
    }
  }

  private CoptsFilter createCoptsFilter(String coptsFilterString) throws EvalException {
    if (Strings.isNullOrEmpty(coptsFilterString)) {
      return CoptsFilter.alwaysPasses();
    } else {
      try {
        return CoptsFilter.fromRegex(Pattern.compile(coptsFilterString));
      } catch (PatternSyntaxException e) {
        throw Starlark.errorf(
            "invalid regular expression '%s': %s", coptsFilterString, e.getMessage());
      }
    }
  }

  @StarlarkMethod(
      name = "create_cc_compile_action_template",
      documented = false,
      parameters = {
        @Param(name = "action_construction_context", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(
            name = "feature_configuration",
            positional = false,
            named = true), // FeatureConfigurationForStarlark
        @Param(name = "compile_build_variables", positional = false, named = true),
        @Param(name = "source", positional = false, named = true),
        @Param(
            name = "additional_compilation_inputs",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "additional_include_scanning_roots",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(name = "use_pic", positional = false, named = true, defaultValue = "False"),
        @Param(name = "output_categories", positional = false, named = true),
        @Param(name = "output_files", positional = false, named = true),
        @Param(name = "dotd_tree_artifact", positional = false, named = true),
        @Param(name = "diagnostics_tree_artifact", positional = false, named = true),
        @Param(name = "lto_indexing_tree_artifact", positional = false, named = true),
        @Param(
            name = "copts_filter",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "None"),
        @Param(name = "needs_include_validation", positional = false, named = true),
        @Param(name = "toolchain_type", positional = false, named = true)
      })
  public void createCppCompileActionTemplate(
      StarlarkRuleContext starlarkRuleContext,
      StarlarkInfo ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      CcToolchainVariables compileBuildVariables,
      Artifact source,
      Sequence<?> additionalCompilationInputs,
      Sequence<?> additionalIncludeScanningRoots,
      boolean usePic,
      Sequence<?> outputCategoriesUnchecked,
      SpecialArtifact outputFiles,
      Object dotdTreeArtifact,
      Object diagnosticsTreeArtifact,
      Object ltoIndexingTreeArtifact,
      Object coptsFilterObject,
      boolean needsIncludeValidation,
      String toolchainType)
      throws RuleErrorException, EvalException {
    CoptsFilter coptsFilter =
        createCoptsFilter(
            Starlark.isNullOrNone(coptsFilterObject) ? null : (String) coptsFilterObject);
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
    ActionOwner owner =
        CppCompileActionBuilder.getActionOwner(starlarkRuleContext.getRuleContext(), toolchainType);
    ImmutableMap<String, String> executionInfo =
        TargetUtils.getExecutionInfo(
            starlarkRuleContext.getRuleContext().getRule(),
            starlarkRuleContext.getRuleContext().isAllowTagsPropagation());
    CppCompileActionBuilder builder =
        createCppCompileActionBuilder(
            owner,
            CcCompilationContext.of(ccCompilationContext),
            ccToolchain,
            configuration,
            coptsFilter,
            featureConfigurationForStarlark,
            source,
            additionalCompilationInputs,
            additionalIncludeScanningRoots,
            outputFiles,
            dotdTreeArtifact,
            diagnosticsTreeArtifact,
            /* gcnoFile= */ null,
            /* dwoFile= */ null,
            /* ltoIndexingFile= */ null,
            usePic,
            needsIncludeValidation,
            executionInfo);
    RuleContext ruleContext = starlarkRuleContext.getRuleContext();
    SpecialArtifact sourceArtifact = (SpecialArtifact) source;
    builder.setVariables(compileBuildVariables);

    try {
      CppCompileActionTemplate actionTemplate =
          new CppCompileActionTemplate(
              sourceArtifact,
              outputFiles,
              CcModule.nullIfNone(dotdTreeArtifact, SpecialArtifact.class),
              CcModule.nullIfNone(diagnosticsTreeArtifact, SpecialArtifact.class),
              CcModule.nullIfNone(ltoIndexingTreeArtifact, SpecialArtifact.class),
              builder,
              CcToolchainProvider.create(ccToolchain),
              outputCategories.build());
      ruleContext.registerAction(actionTemplate);
    } catch (EvalException e) {
      throw new RuleErrorException(e.getMessage());
    }
  }

  private static CppCompileActionBuilder createCppCompileActionBuilder(
      ActionOwner owner,
      CcCompilationContext ccCompilationContext,
      StarlarkInfo ccToolchain,
      BuildConfigurationValue configuration,
      CoptsFilter coptsFilter,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      Artifact sourceArtifact,
      Sequence<?> additionalCompilationInputs,
      Sequence<?> additionalIncludeScanningRoots,
      Object outputFile,
      Object dotdFile,
      Object diagnosticsFile,
      Object gcnoFile,
      Object dwoFile,
      Object ltoIndexingFile,
      boolean usePic,
      boolean needsIncludeValidation,
      ImmutableMap<String, String> executionInfo)
      throws EvalException {
    CppCompileActionBuilder builder =
        new CppCompileActionBuilder(owner, CcToolchainProvider.create(ccToolchain), configuration)
            .setSourceFile(sourceArtifact)
            .setCcCompilationContext(ccCompilationContext)
            .setCoptsFilter(coptsFilter)
            .setFeatureConfiguration(featureConfigurationForStarlark.getFeatureConfiguration())
            .addExecutionInfo(executionInfo);
    if (additionalCompilationInputs.size() > 0) {
      builder.addMandatoryInputs(
          Sequence.cast(
              additionalCompilationInputs, Artifact.class, "additional_compilation_inputs"));
    }
    if (additionalIncludeScanningRoots.size() > 0) {
      builder.addAdditionalIncludeScanningRoots(
          Sequence.cast(
              additionalIncludeScanningRoots, Artifact.class, "additional_include_scanning_roots"));
    }
    builder.setGcnoFile(nullIfNone(gcnoFile, Artifact.class));
    builder.setDwoFile(nullIfNone(dwoFile, Artifact.class));
    builder.setLtoIndexingFile(nullIfNone(ltoIndexingFile, Artifact.class));
    builder.setOutputs(
        nullIfNone(outputFile, Artifact.class),
        nullIfNone(dotdFile, Artifact.class),
        nullIfNone(diagnosticsFile, Artifact.class));
    builder.setPicMode(usePic);
    builder.setNeedsIncludeValidation(needsIncludeValidation);
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

  @StarlarkMethod(
      name = "per_file_copts",
      documented = false,
      parameters = {
        @Param(name = "cpp_configuration"),
        @Param(name = "source_file"),
        @Param(name = "label"),
      })
  public ImmutableList<String> perFileCopts(
      CppConfiguration cppConfiguration, Artifact sourceFile, Label sourceLabel)
      throws EvalException {
    return cppConfiguration.getPerFileCopts().stream()
        .filter(
            perLabelOptions ->
                (sourceLabel != null && perLabelOptions.isIncluded(sourceLabel))
                    || perLabelOptions.isIncluded(sourceFile))
        .map(PerLabelOptions::getOptions)
        .flatMap(options -> options.stream())
        .collect(toImmutableList());
  }

  @StarlarkMethod(
      name = "declare_compile_output_file",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "label", positional = false, named = true),
        @Param(name = "output_name", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
      })
  public Artifact declareCompileOutputFile(
      StarlarkRuleContext starlarkRuleContext,
      Label label,
      String outputName,
      BuildConfigurationValue configuration) {
    RuleContext ruleContext = starlarkRuleContext.getRuleContext();
    return CppHelper.getCompileOutputArtifact(ruleContext, label, outputName, configuration);
  }

  @StarlarkMethod(
      name = "declare_other_output_file",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "output_name", positional = false, named = true),
        @Param(name = "object_file", positional = false, named = true)
      })
  public Artifact declareOtherOutputFile(
      StarlarkRuleContext starlarkRuleContext, String outputName, Artifact objectFile) {
    RuleContext ruleContext = starlarkRuleContext.getRuleContext();
    return ruleContext.getDerivedArtifact(
        objectFile.getRootRelativePath().getParentDirectory().getRelative(outputName),
        ruleContext.getConfiguration().getBinDirectory(ruleContext.getLabel().getRepository()));
  }

  @StarlarkMethod(
      name = "create_lto_backend_action",
      documented = false,
      parameters = {
        @Param(name = "actions", positional = false, named = true),
        @Param(name = "feature_configuration", positional = false, named = true),
        @Param(name = "build_variables", positional = false, named = true),
        @Param(name = "use_pic", positional = false, named = true),
        @Param(name = "inputs", positional = false, named = true),
        @Param(
            name = "all_bitcode_files",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Depset.class), @ParamType(type = NoneType.class)}),
        @Param(
            name = "imports",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Artifact.class), @ParamType(type = NoneType.class)}),
        @Param(name = "outputs", positional = false, named = true),
        @Param(name = "env", positional = false, named = true),
      })
  public void createLtoBackendAction(
      StarlarkActionFactory actions,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      CcToolchainVariables buildVariables,
      boolean usePic,
      Depset inputs,
      Object allBitcodeFiles,
      Object imports,
      Sequence<?> outputs,
      Dict<?, ?> env)
      throws EvalException {
    FeatureConfiguration featureConfiguration =
        featureConfigurationForStarlark.getFeatureConfiguration();
    BitcodeFiles bitcodeFiles =
        allBitcodeFiles == Starlark.NONE
            ? null
            : new BitcodeFiles(Depset.cast(allBitcodeFiles, Artifact.class, "bitcode_files"));
    LtoBackendAction action =
        LtoBackendArtifacts.createLtoBackendActionForStarlark(
            actions.getRuleContext().getActionOwner(),
            actions.getRuleContext().getConfiguration(),
            featureConfiguration,
            buildVariables,
            usePic,
            Depset.cast(inputs, Artifact.class, "inputs"),
            bitcodeFiles,
            imports instanceof Artifact importsArtifact ? importsArtifact : null,
            ImmutableSet.copyOf(Sequence.cast(outputs, Artifact.class, "outputs")),
            ActionEnvironment.create(
                ImmutableMap.copyOf(Dict.cast(env, String.class, String.class, "env"))));
    actions.getRuleContext().registerAction(action);
  }

  @StarlarkMethod(
      name = "create_lto_backend_action_template",
      documented = false,
      parameters = {
        @Param(name = "actions", positional = false, named = true),
        @Param(name = "feature_configuration", positional = false, named = true),
        @Param(name = "build_variables", positional = false, named = true),
        @Param(name = "use_pic", positional = false, named = true),
        @Param(
            name = "all_bitcode_files",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Depset.class), @ParamType(type = NoneType.class)}),
        @Param(name = "additional_inputs", positional = false, named = true),
        @Param(
            name = "index",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SpecialArtifact.class),
              @ParamType(type = NoneType.class)
            }),
        @Param(
            name = "bitcode_file",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SpecialArtifact.class),
              @ParamType(type = NoneType.class)
            }),
        @Param(
            name = "object_file",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SpecialArtifact.class),
              @ParamType(type = NoneType.class)
            }),
        @Param(
            name = "dwo_file",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SpecialArtifact.class),
              @ParamType(type = NoneType.class)
            }),
        @Param(name = "env", positional = false, named = true),
      })
  public void createLtoBackendActionTemplate(
      StarlarkActionFactory actions,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      CcToolchainVariables buildVariables,
      boolean usePic,
      Object allBitcodeFiles,
      Depset additionalInputs,
      Object indexObj,
      Object bitcodeFileObj,
      Object objectFileObj,
      Object dwoFileObj,
      Dict<?, ?> env)
      throws EvalException {
    FeatureConfiguration featureConfiguration =
        featureConfigurationForStarlark.getFeatureConfiguration();
    BitcodeFiles bitcodeFiles =
        allBitcodeFiles == Starlark.NONE
            ? null
            : new BitcodeFiles(Depset.cast(allBitcodeFiles, Artifact.class, "bitcode_files"));
    LtoBackendActionTemplate actionTemplate =
        new LtoBackendActionTemplate(
            indexObj instanceof SpecialArtifact index ? index : null,
            bitcodeFileObj instanceof SpecialArtifact bitcodeFile ? bitcodeFile : null,
            objectFileObj instanceof SpecialArtifact objectFile ? objectFile : null,
            dwoFileObj instanceof SpecialArtifact dwoFile ? dwoFile : null,
            featureConfiguration,
            Depset.cast(additionalInputs, Artifact.class, "additional_inputs"),
            ActionEnvironment.create(
                ImmutableMap.copyOf(Dict.cast(env, String.class, String.class, "env"))),
            buildVariables,
            usePic,
            bitcodeFiles,
            actions.getRuleContext().getActionOwner());
    actions.getRuleContext().registerAction(actionTemplate);
  }

  @StarlarkMethod(
      name = "create_header_info",
      documented = false,
      parameters = {
        @Param(name = "header_module", positional = false, named = true, defaultValue = "None"),
        @Param(name = "pic_header_module", positional = false, named = true, defaultValue = "None"),
        @Param(
            name = "modular_public_headers",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(
            name = "modular_private_headers",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(name = "textual_headers", positional = false, named = true, defaultValue = "[]"),
        @Param(
            name = "separate_module_headers",
            positional = false,
            named = true,
            defaultValue = "[]"),
        @Param(name = "separate_module", positional = false, named = true, defaultValue = "None"),
        @Param(
            name = "separate_pic_module",
            positional = false,
            named = true,
            defaultValue = "None"),
      },
      useStarlarkThread = true)
  public HeaderInfo createHeaderInfo(
      Object headerModule,
      Object picHeaderModule,
      Sequence<?> modularPublicHeaders,
      Sequence<?> modularPrivateHeaders,
      Sequence<?> textualHeaders,
      Sequence<?> separateModuleHeaders,
      Object separateModule,
      Object separatePicModule,
      StarlarkThread thread)
      throws EvalException {
    return HeaderInfo.create(
        thread.getNextIdentityToken(),
        nullIfNone(headerModule, Artifact.DerivedArtifact.class),
        nullIfNone(picHeaderModule, Artifact.DerivedArtifact.class),
        Sequence.cast(modularPublicHeaders, Artifact.class, "modular_public_headers")
            .getImmutableList(),
        Sequence.cast(modularPrivateHeaders, Artifact.class, "modular_private_headers")
            .getImmutableList(),
        Sequence.cast(textualHeaders, Artifact.class, "textual_headers").getImmutableList(),
        Sequence.cast(separateModuleHeaders, Artifact.class, "separate_module_headers")
            .getImmutableList(),
        nullIfNone(separateModule, Artifact.DerivedArtifact.class),
        nullIfNone(separatePicModule, Artifact.DerivedArtifact.class),
        ImmutableList.of(),
        ImmutableList.of());
  }

  @StarlarkMethod(
      name = "create_header_info_with_deps",
      documented = false,
      parameters = {
        @Param(name = "header_info", positional = false, named = true, defaultValue = "None"),
        @Param(name = "deps", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "merged_deps", positional = false, named = true, defaultValue = "[]")
      },
      useStarlarkThread = true)
  public HeaderInfo createHeaderInfoWithDeps(
      HeaderInfo headerInfo, Sequence<?> deps, Sequence<?> mergedDeps, StarlarkThread thread)
      throws EvalException {
    return HeaderInfo.create(
        thread.getNextIdentityToken(),
        headerInfo.headerModule,
        headerInfo.picHeaderModule,
        headerInfo.modularPublicHeaders,
        headerInfo.modularPrivateHeaders,
        headerInfo.textualHeaders,
        headerInfo.separateModuleHeaders,
        headerInfo.separateModule,
        headerInfo.separatePicModule,
        Sequence.cast(deps, HeaderInfo.class, "deps").getImmutableList(),
        Sequence.cast(mergedDeps, HeaderInfo.class, "merged_deps").getImmutableList());
  }

  /**
   * Run variable expansion and shell tokenization on a sequence of flags.
   *
   * <p>When expanding path variables (e.g. $(execpath ...)), the label can refer to any of which in
   * the {@code srcs}, {@code non_arc_srcs}, {@code hdrs} or {@code data} attributes or an output of
   * the target.
   *
   * @param starlarkRuleContext The rule context of the expansion.
   * @param attributeName The attribute of the rule tied to the expansion. Used for error reporting
   *     only.
   * @param flags The sequence of flags to expand.
   */
  @StarlarkMethod(
      name = "expand_and_tokenize",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "attr", positional = false, named = true),
        @Param(name = "flags", positional = false, defaultValue = "[]", named = true),
      })
  public Sequence<String> expandAndTokenize(
      StarlarkRuleContext starlarkRuleContext, String attributeName, Sequence<?> flags)
      throws EvalException, InterruptedException {
    if (flags.isEmpty()) {
      return Sequence.cast(flags, String.class, attributeName);
    }
    Expander expander =
        starlarkRuleContext
            .getRuleContext()
            .getExpander(
                StarlarkRuleContext.makeLabelMap(
                    ImmutableSet.copyOf(
                        Iterables.concat(
                            starlarkRuleContext.getRuleContext().getPrerequisites("srcs"),
                            starlarkRuleContext.getRuleContext().getPrerequisites("non_arc_srcs"),
                            starlarkRuleContext.getRuleContext().getPrerequisites("hdrs"),
                            starlarkRuleContext.getRuleContext().getPrerequisites("data"),
                            starlarkRuleContext
                                .getRuleContext()
                                .getPrerequisites("additional_linker_inputs"))),
                    starlarkRuleContext
                        .getStarlarkSemantics()
                        .getBool(BuildLanguageOptions.INCOMPATIBLE_LOCATIONS_PREFERS_EXECUTABLE)))
            .withDataExecLocations();
    ImmutableList<String> expandedFlags =
        expander.tokenized(attributeName, Sequence.cast(flags, String.class, attributeName));
    return StarlarkList.immutableCopyOf(expandedFlags);
  }
}
