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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.Type;
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
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.Location;

/** Utility methods for rules in Starlark Builtins */
@StarlarkBuiltin(name = "cc_internal", category = DocCategory.BUILTIN, documented = false)
public class CcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "cc_internal";

  @Nullable
  private PathFragment getPathfragmentOrNone(Object o) {
    String pathString = CcModule.convertFromNoneable(o, null);
    if (pathString == null) {
      return null;
    }
    return PathFragment.create(pathString);
  }

  private ImmutableMap<String, PathFragment> castDict(Dict<?, ?> d) throws EvalException {
    return Dict.cast(d, String.class, String.class, "tool_paths").entrySet().stream()
        .map(p -> Pair.of(p.getKey(), PathFragment.create(p.getValue())))
        .collect(toImmutableMap(Pair::getFirst, Pair::getSecond));
  }

  @StarlarkMethod(
      name = "construct_cc_toolchain_attributes_info",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "is_apple", positional = false, named = true),
        @Param(name = "build_vars_func", positional = false, named = true),
      })
  public CcToolchainAttributesProvider constructCcToolchainAttributesInfo(
      StarlarkRuleContext ruleContext, boolean isApple, Object buildVarsFunc) throws EvalException {
    return new CcToolchainAttributesProvider(
        ruleContext.getRuleContext(), isApple, (StarlarkFunction) buildVarsFunc);
  }

  @StarlarkMethod(
      name = "construct_toolchain_provider",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "cpp_config", positional = false, named = true),
        @Param(name = "toolchain_features", positional = false, named = true),
        @Param(name = "tools_directory", positional = false, named = true),
        @Param(name = "attributes", positional = false, named = true),
        @Param(
            name = "static_runtime_link_inputs",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Depset.class),
              @ParamType(type = NoneType.class),
            }),
        @Param(
            name = "dynamic_runtime_link_symlinks",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = Depset.class),
              @ParamType(type = NoneType.class),
            }),
        @Param(name = "runtime_solib_dir", positional = false, named = true),
        @Param(name = "cc_compilation_context", positional = false, named = true),
        @Param(name = "builtin_include_files", positional = false, named = true),
        @Param(name = "target_builtin_include_files", positional = false, named = true),
        @Param(name = "builtin_include_directories", positional = false, named = true),
        @Param(name = "sysroot", positional = false, named = true),
        @Param(name = "target_sysroot", positional = false, named = true),
        @Param(name = "fdo_context", positional = false, named = true),
        @Param(name = "is_tool_configuration", positional = false, named = true),
        @Param(name = "tool_paths", positional = false, named = true),
        @Param(name = "toolchain_config_info", positional = false, named = true),
        @Param(name = "default_sysroot", positional = false, named = true),
        @Param(name = "runtime_sysroot", positional = false, named = true),
        @Param(name = "solib_directory", positional = false, named = true),
        @Param(name = "additional_make_variables", positional = false, named = true),
        @Param(name = "legacy_cc_flags_make_variable", positional = false, named = true),
        @Param(name = "objcopy", positional = false, named = true),
        @Param(name = "compiler", positional = false, named = true),
        @Param(name = "preprocessor", positional = false, named = true),
        @Param(name = "nm", positional = false, named = true),
        @Param(name = "objdump", positional = false, named = true),
        @Param(name = "ar", positional = false, named = true),
        @Param(name = "strip", positional = false, named = true),
        @Param(name = "ld", positional = false, named = true),
        @Param(name = "gcov", positional = false, named = true),
        @Param(name = "vars", positional = false, named = true),
      })
  public CcToolchainProvider getCcToolchainProvider(
      StarlarkRuleContext ruleContext,
      Object cppConfigurationObject,
      CcToolchainFeatures toolchainFeatures,
      String toolsDirectoryStr,
      CcToolchainAttributesProvider attributes,
      Object staticRuntimeLinkInputsObject,
      Object dynamicRuntimeLinkInputsObject,
      String dynamicRuntimeSolibDirStr,
      CcCompilationContext ccCompilationContext,
      Sequence<?> builtinIncludeFiles,
      Sequence<?> targetBuiltinIncludeFiles,
      Sequence<?> builtInIncludeDirectoriesStr,
      Object sysrootObject,
      Object targetSysrootObject,
      FdoContext fdoContext,
      boolean isToolConfiguration,
      Dict<?, ?> toolPathsDict,
      CcToolchainConfigInfo toolchainConfigInfo,
      Object defaultSysrootObject,
      Object runtimeSysrootObject,
      String solibDirectory,
      Dict<?, ?> additionalMakeVariablesDict,
      String legacyCcFlagsMakeVariable,
      String objcopyExecutable,
      String compilerExecutable,
      String preprocessorExecutable,
      String nmExecutable,
      String objdumpExecutable,
      String arExecutable,
      String stripExecutable,
      String ldExecutable,
      String gcovExecutable,
      Object vars)
      throws EvalException {
    CppConfiguration cppConfiguration = CcModule.convertFromNoneable(cppConfigurationObject, null);
    PathFragment toolsDirectory = PathFragment.create(toolsDirectoryStr);
    NestedSet<Artifact> staticRuntimeLinkInputsSet = null;
    NestedSet<Artifact> dynamicRuntimeLinkInputsSet = null;
    try {
      if (staticRuntimeLinkInputsObject != Starlark.NONE) {
        staticRuntimeLinkInputsSet =
            ((Depset) staticRuntimeLinkInputsObject).getSet(Artifact.class);
      }
      if (dynamicRuntimeLinkInputsObject != Starlark.NONE) {
        dynamicRuntimeLinkInputsSet =
            ((Depset) dynamicRuntimeLinkInputsObject).getSet(Artifact.class);
      }
    } catch (TypeException e) {
      throw new EvalException(e);
    }
    PathFragment dynamicRuntimeSolibDir = PathFragment.create(dynamicRuntimeSolibDirStr);
    ImmutableList<PathFragment> builtInIncludeDirectories =
        Sequence.cast(builtInIncludeDirectoriesStr, String.class, "builtin_include_directories")
            .stream()
            .map(PathFragment::create)
            .collect(toImmutableList());
    PathFragment sysroot = getPathfragmentOrNone(sysrootObject);
    PathFragment targetSysroot = getPathfragmentOrNone(targetSysrootObject);
    Dict<String, String> additionalMakeVariables =
        Dict.cast(additionalMakeVariablesDict, String.class, String.class, "tool_paths");
    PathFragment defaultSysroot = getPathfragmentOrNone(defaultSysrootObject);
    PathFragment runtimeSysroot = getPathfragmentOrNone(runtimeSysrootObject);

    return new CcToolchainProvider(
        /* cppConfiguration= */ cppConfiguration,
        /* toolchainFeatures= */ toolchainFeatures,
        /* crosstoolTopPathFragment= */ toolsDirectory,
        /* allFiles= */ attributes.getAllFiles(),
        /* allFilesIncludingLibc= */ attributes.getFullInputsForCrosstool(),
        /* compilerFiles= */ attributes.getCompilerFiles(),
        /* compilerFilesWithoutIncludes= */ attributes.getCompilerFilesWithoutIncludes(),
        /* stripFiles= */ attributes.getStripFiles(),
        /* objcopyFiles= */ attributes.getObjcopyFiles(),
        /* asFiles= */ attributes.getAsFiles(),
        /* arFiles= */ attributes.getArFiles(),
        /* linkerFiles= */ attributes.getFullInputsForLink(),
        /* interfaceSoBuilder= */ attributes.getIfsoBuilder(),
        /* dwpFiles= */ attributes.getDwpFiles(),
        /* coverageFiles= */ attributes.getCoverage(),
        /* libcLink= */ attributes.getLibc(),
        /* targetLibcLink= */ attributes.getTargetLibc(),
        /* staticRuntimeLinkInputs= */ staticRuntimeLinkInputsSet,
        /* dynamicRuntimeLinkInputs= */ dynamicRuntimeLinkInputsSet,
        /* dynamicRuntimeSolibDir= */ dynamicRuntimeSolibDir,
        /* ccCompilationContext= */ ccCompilationContext,
        /* supportsParamFiles= */ attributes.isSupportsParamFiles(),
        /* supportsHeaderParsing= */ attributes.isSupportsHeaderParsing(),
        /* buildOptions */ ruleContext.getRuleContext().getConfiguration().getOptions(),
        /* buildVariables= */ (CcToolchainVariables) vars,
        /* builtinIncludeFiles= */ Sequence.cast(
                builtinIncludeFiles, Artifact.class, "builtin_include_files")
            .getImmutableList(),
        /* targetBuiltinIncludeFiles= */ Sequence.cast(
                targetBuiltinIncludeFiles, Artifact.class, "target_builtin_include_files")
            .getImmutableList(),
        /* linkDynamicLibraryTool= */ attributes.getLinkDynamicLibraryTool(),
        /* grepIncludes= */ attributes.getGrepIncludes(),
        /* builtInIncludeDirectories= */ builtInIncludeDirectories,
        /* sysroot= */ sysroot,
        /* targetSysroot= */ targetSysroot,
        /* fdoContext= */ fdoContext,
        /* isToolConfiguration= */ isToolConfiguration,
        /* licensesProvider= */ attributes.getLicensesProvider(),
        /* toolPaths= */ castDict(toolPathsDict),
        /* toolchainIdentifier= */ toolchainConfigInfo.getToolchainIdentifier(),
        /* compiler= */ toolchainConfigInfo.getCompiler(),
        /* abiGlibcVersion= */ toolchainConfigInfo.getAbiLibcVersion(),
        /* targetCpu= */ toolchainConfigInfo.getTargetCpu(),
        /* targetOS= */ toolchainConfigInfo.getCcTargetOs(),
        /* defaultSysroot= */ defaultSysroot,
        /* runtimeSysroot= */ runtimeSysroot,
        /* targetLibc= */ toolchainConfigInfo.getTargetLibc(),
        /* ccToolchainLabel= */ ruleContext.getRuleContext().getLabel(),
        /* solibDirectory= */ solibDirectory,
        /* abi= */ toolchainConfigInfo.getAbiVersion(),
        /* targetSystemName= */ toolchainConfigInfo.getTargetSystemName(),
        /* additionalMakeVariables= */ ImmutableMap.copyOf(additionalMakeVariables),
        /* legacyCcFlagsMakeVariable= */ legacyCcFlagsMakeVariable,
        /* allowlistForLayeringCheck= */ attributes.getAllowlistForLayeringCheck(),
        /* allowListForLooseHeaderCheck= */ attributes.getAllowlistForLooseHeaderCheck(),
        /* objcopyExecutable= */ objcopyExecutable,
        /* compilerExecutable= */ compilerExecutable,
        /* preprocessorExecutable= */ preprocessorExecutable,
        /* nmExecutable= */ nmExecutable,
        /* objdumpExecutable= */ objdumpExecutable,
        /* arExecutable= */ arExecutable,
        /* stripExecutable= */ stripExecutable,
        /* ldExecutable= */ ldExecutable,
        /* gcovExecutable= */ gcovExecutable,
        /* ccToolchainBuildVariablesFunc */ attributes.getCcToolchainBuildVariablesFunc());
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
        @Param(name = "attributes", positional = false, named = true),
        @Param(name = "configuration", positional = false, named = true),
        @Param(name = "cpp_config", positional = false, named = true),
        @Param(name = "tool_paths", positional = false, named = true),
      },
      allowReturnNones = true)
  @Nullable
  public FdoContext fdoContext(
      StarlarkRuleContext ruleContext,
      CcToolchainAttributesProvider attributes,
      BuildConfigurationValue configuration,
      CppConfiguration cppConfiguration,
      Dict<?, ?> toolPathsDict)
      throws EvalException, InterruptedException {
    try {
      return FdoHelper.getFdoContext(
          ruleContext.getRuleContext(),
          attributes,
          configuration,
          cppConfiguration,
          castDict(toolPathsDict));
    } catch (RuleErrorException e) {
      throw new EvalException(e);
    }
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
  public CcCommon createCommon(StarlarkRuleContext starlarkRuleContext) throws EvalException {
    try {
      return new CcCommon(starlarkRuleContext.getRuleContext());
    } catch (RuleErrorException e) {
      throw new EvalException(e);
    }
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
          starlarkActionFactoryApi.getActionConstructionContext().getActionKeyContext());
    } catch (CommandLineExpansionException | InterruptedException ex) {
      throw new EvalException(ex);
    }
  }

  static class DefaultHdrsCheckBuiltinComputedDefault extends ComputedDefault
      implements NativeComputedDefaultApi {
    @Override
    public Object getDefault(AttributeMap rule) {
      return rule.getPackageArgs().isDefaultHdrsCheckSet()
          ? rule.getPackageArgs().getDefaultHdrsCheck()
          : "";
    }

    @Override
    public boolean resolvableWithRawAttributes() {
      return true;
    }
  }

  @StarlarkMethod(name = "default_hdrs_check_computed_default", documented = false)
  public ComputedDefault getDefaultHdrsCheckComputedDefault() {
    return new DefaultHdrsCheckBuiltinComputedDefault();
  }

  /**
   * TODO(bazel-team): This can be re-written directly to Starlark but it will cause a memory
   * regression due to the way StarlarkComputedDefault is stored for each rule.
   */
  static class StlComputedDefault extends ComputedDefault implements NativeComputedDefaultApi {
    @Override
    @Nullable
    public Object getDefault(AttributeMap rule) {
      return rule.getOrDefault("tags", Type.STRING_LIST, ImmutableList.of()).contains("__CC_STL__")
          ? null
          : Label.parseCanonicalUnchecked("@//third_party/stl");
    }

    @Override
    public boolean resolvableWithRawAttributes() {
      return true;
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
}
