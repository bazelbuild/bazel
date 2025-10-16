// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext.HeaderInfo;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcModuleApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/**
 * A module that contains Starlark utilities for C++ support.
 *
 * <p>The Bazel team is planning to rewrite all native rules in Starlark. Many of these rules use
 * C++ functionality that is not presently exposed to the public Starlark C++ API. To speed up the
 * transition to Starlark, we are exposing functionality "as is" but preventing its use externally
 * until we are comfortable with the API which would need to be supported long term.
 *
 * <p>We are not opposed to gradually adding to and improving the public C++ API but nothing should
 * merged without following proper design processes and discussions.
 */
public abstract class CcModule
    implements CcModuleApi<
        StarlarkActionFactory,
        Artifact,
        FeatureConfigurationForStarlark,
        CcCompilationContext,
        CcToolchainVariables,
        ConstraintValueInfo,
        StarlarkRuleContext,
        CppModuleMap> {

  public abstract CppSemantics getSemantics(Language language);

  @Override
  public Provider getCcToolchainProvider() {
    // TODO: b/433485282 this will need to change for Bazel once we update rules_cc containing
    // cl/791606702
    return CcToolchainProvider.BUILTINS_PROVIDER;
  }

  @Override
  public String getToolForAction(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    try {
      return featureConfiguration.getFeatureConfiguration().getToolPathForAction(actionName);
    } catch (IllegalArgumentException illegalArgumentException) {
      throw new EvalException(illegalArgumentException);
    }
  }

  // TODO(blaze-team): duplicate with the getExecutionRequirements below
  @Override
  public Sequence<String> getToolRequirementForAction(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return StarlarkList.immutableCopyOf(
        featureConfiguration.getFeatureConfiguration().getToolRequirementsForAction(actionName));
  }

  @Override
  public Sequence<String> getExecutionRequirements(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return StarlarkList.immutableCopyOf(
        featureConfiguration.getFeatureConfiguration().getToolRequirementsForAction(actionName));
  }

  @Override
  public boolean actionIsEnabled(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return featureConfiguration.getFeatureConfiguration().actionIsConfigured(actionName);
  }

  @Override
  public Sequence<String> getCommandLine(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      CcToolchainVariables variables,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return StarlarkList.immutableCopyOf(
        featureConfiguration.getFeatureConfiguration().getCommandLine(actionName, variables));
  }

  @Override
  public Dict<String, String> getEnvironmentVariable(
      FeatureConfigurationForStarlark featureConfiguration,
      String actionName,
      CcToolchainVariables variables,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return Dict.immutableCopyOf(
        featureConfiguration
            .getFeatureConfiguration()
            .getEnvironmentVariables(actionName, variables, PathMapper.NOOP));
  }

  @Override
  public CcToolchainVariables getVariables(StarlarkThread thread) throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return CcToolchainVariables.empty();
  }

  /**
   * Converts an object that can be the NoneType to the actual object if it is not or returns the
   * default value if none.
   *
   * <p>This operation is wildly unsound. It performs no dymamic checks (casts), it simply lies
   * about the type.
   */
  @SuppressWarnings("unchecked")
  protected static <T> T convertFromNoneable(Object obj, @Nullable T defaultValue) {
    if (Starlark.UNBOUND == obj || Starlark.isNullOrNone(obj)) {
      return defaultValue;
    }
    return (T) obj; // totally unsafe
  }

  @Override
  public CcCompilationContext createCcCompilationContext(
      Object headers,
      Object systemIncludes,
      Object includes,
      Object quoteIncludes,
      Object frameworkIncludes,
      Object defines,
      Object localDefines,
      Sequence<?> directTextualHdrs,
      Sequence<?> directPublicHdrs,
      Sequence<?> directPrivateHdrs,
      Object moduleMap,
      Object externalIncludes,
      Object virtualToOriginalHeaders,
      Sequence<?> dependentCcCompilationContexts,
      Sequence<?> exportedDependentCcCompilationContexts,
      Sequence<?> nonCodeInputs,
      Sequence<?> looseHdrsDirsObject,
      String headersCheckingMode,
      Object picHeaderModule,
      Object headerModule,
      Sequence<?> separateModuleHeaders,
      Object separateModule,
      Object separatePicModule,
      Object addPublicHeadersToModularHeaders,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);

    // Public parameters.
    ImmutableList<Artifact> headerList = toNestedSetOfArtifacts(headers, "headers").toList();
    ImmutableList<Artifact> textualHdrsList =
        Sequence.cast(directTextualHdrs, Artifact.class, "direct_textual_headers")
            .getImmutableList();
    ImmutableList.Builder<Artifact> modularPublicHdrsList = ImmutableList.builder();
    modularPublicHdrsList.addAll(
        Sequence.cast(directPublicHdrs, Artifact.class, "direct_public_headers"));
    ImmutableList<Artifact> modularPrivateHdrsList =
        Sequence.cast(directPrivateHdrs, Artifact.class, "direct_private_headers")
            .getImmutableList();

    if ((Boolean) addPublicHeadersToModularHeaders) {
      modularPublicHdrsList.addAll(headerList);
    }

    HeaderInfo headerInfo =
        HeaderInfo.create(
            thread.getNextIdentityToken(),
            // TODO(ilist@): typechecks; user code can throw ClassCastException
            headerModule == Starlark.NONE ? null : (Artifact.DerivedArtifact) headerModule,
            picHeaderModule == Starlark.NONE ? null : (Artifact.DerivedArtifact) picHeaderModule,
            modularPublicHdrsList.build(),
            modularPrivateHdrsList,
            textualHdrsList,
            Sequence.cast(separateModuleHeaders, Artifact.class, "separate_module_headers")
                .getImmutableList(),
            convertFromNoneable(separateModule, null),
            convertFromNoneable(separatePicModule, null),
            ImmutableList.of(),
            ImmutableList.of());

    CcCompilationContext single =
        CcCompilationContext.create(
            new CcCompilationContext.CommandLineCcCompilationContext(
                toPathFragments(includes, "includes"),
                toPathFragments(quoteIncludes, "quote_includes"),
                toPathFragments(systemIncludes, "system_includes"),
                toPathFragments(frameworkIncludes, "framework_includes"),
                toPathFragments(externalIncludes, "external_includes"),
                toNestedSetOfStrings(defines, "defines").toList(),
                toNestedSetOfStrings(localDefines, "local_defines").toList()),
            /* declaredIncludeSrcs= */ NestedSetBuilder.wrap(Order.STABLE_ORDER, headerList),
            NestedSetBuilder.wrap(
                Order.STABLE_ORDER,
                Sequence.cast(nonCodeInputs, Artifact.class, "non_code_inputs")),
            headerInfo,
            /* transitiveModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /* transitivePicModules= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /* directModuleMaps= */ ImmutableList.of(),
            /* exportingModuleMaps= */ ImmutableList.of(),
            moduleMap instanceof CppModuleMap cppModuleMap ? cppModuleMap : null,
            Depset.cast(virtualToOriginalHeaders, Tuple.class, "virtual_to_original_headers"),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER));

    Sequence<CcCompilationContext> exportedDeps =
        Sequence.cast(
            exportedDependentCcCompilationContexts,
            CcCompilationContext.class,
            "exported_dependent_cc_compilation_contexts");
    Sequence<CcCompilationContext> deps =
        Sequence.cast(
            dependentCcCompilationContexts,
            CcCompilationContext.class,
            "dependent_cc_compilation_contexts");

    return CcCompilationContext.createAndMerge(
        thread.getNextIdentityToken(), single, exportedDeps, deps);
  }

  @Override
  public CcCompilationContext mergeCompilationContexts(
      Sequence<?> compilationContexts,
      Sequence<?> nonExportedCompilationContexts,
      StarlarkThread thread)
      throws EvalException {
    if (compilationContexts.isEmpty() && nonExportedCompilationContexts.isEmpty()) {
      return CcCompilationContext.EMPTY;
    }

    Sequence<CcCompilationContext> exportedDeps =
        Sequence.cast(compilationContexts, CcCompilationContext.class, "compilation_contexts");
    Sequence<CcCompilationContext> deps =
        Sequence.cast(
            nonExportedCompilationContexts,
            CcCompilationContext.class,
            "non_exported_compilation_contexts");

    return CcCompilationContext.createAndMerge(
        thread.getNextIdentityToken(), CcCompilationContext.EMPTY, exportedDeps, deps);
  }

  private static NestedSet<Artifact> toNestedSetOfArtifacts(Object obj, String fieldName)
      throws EvalException {
    if (obj == Starlark.UNBOUND) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    } else {
      return Depset.noneableCast(obj, Artifact.class, fieldName);
    }
  }

  private static NestedSet<String> toNestedSetOfStrings(Object obj, String fieldName)
      throws EvalException {
    if (obj == Starlark.UNBOUND) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    } else {
      return Depset.noneableCast(obj, String.class, fieldName);
    }
  }

  private static ImmutableList<PathFragment> toPathFragments(Object obj, String fieldName)
      throws EvalException {
    return toNestedSetOfStrings(obj, fieldName).toList().stream()
        .map(x -> PathFragment.create(x))
        .collect(toImmutableList());
  }

  @Override
  public CppModuleMap createCppModuleMap(Artifact file, String name, StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return new CppModuleMap(file, name);
  }

  @Override
  public boolean checkExperimentalCcSharedLibrary(StarlarkThread thread) throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return thread.getSemantics().getBool(BuildLanguageOptions.EXPERIMENTAL_CC_SHARED_LIBRARY);
  }

  @Override
  public boolean getIncompatibleDisableObjcLibraryTransition(StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return thread
        .getSemantics()
        .getBool(BuildLanguageOptions.INCOMPATIBLE_DISABLE_OBJC_LIBRARY_TRANSITION);
  }

  @Override
  public boolean addGoExecGroupsToBinaryRules(StarlarkThread thread) {
    return thread.getSemantics().getBool(BuildLanguageOptions.ADD_GO_EXEC_GROUPS_TO_BINARY_RULES);
  }

  // TODO(b/65151735): Remove when cc_flags is entirely from features.
  @Override
  public String legacyCcFlagsMakeVariable(Info ccToolchainInfo, StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    CcToolchainProvider ccToolchain = CcToolchainProvider.wrapOrThrowEvalException(ccToolchainInfo);
    return ccToolchain.getLegacyCcFlagsMakeVariable();
  }

  @Nullable
  static <T> T nullIfNone(Object object, Class<T> type) {
    return object != Starlark.NONE ? type.cast(object) : null;
  }

  public static void checkPrivateStarlarkificationAllowlist(StarlarkThread thread)
      throws EvalException {
    BuiltinRestriction.failIfCalledOutsideDefaultAllowlist(thread);
  }

  public static boolean isStarlarkCcCommonCalledFromBuiltins(StarlarkThread thread) {
    Label label =
        ((BazelModuleContext)
                Module.ofInnermostEnclosingStarlarkFunction(thread, 1).getClientData())
            .label();
    return label.getPackageIdentifier().getRepository().getName().equals("_builtins");
  }

  protected static void isCalledFromStarlarkCcCommon(StarlarkThread thread) throws EvalException {
    Label label = BazelModuleContext.ofInnermostBzlOrThrow(thread).label();
    // Allow direct access to cc_common.bzl and to C++ linking code that can't use cc_common.bzl
    // directly without creating a cycle.
    if (!label.getCanonicalForm().endsWith("_builtins//:common/cc/cc_common.bzl")
        && !label.getCanonicalForm().contains("_builtins//:common/cc/compile")
        && !label.getCanonicalForm().contains("_builtins//:common/cc/link")
        && !label.getCanonicalForm().contains("_builtins//:common/cc/toolchain_config")) {
      throw Starlark.errorf(
          "cc_common_internal can only be used by cc_common.bzl in builtins, "
              + "please use cc_common instead.");
    }
  }

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

  protected Language parseLanguage(String string) throws EvalException {
    try {
      return Language.valueOf(Ascii.toUpperCase(string.replace('+', 'p')));
    } catch (IllegalArgumentException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
  }

  @StarlarkMethod(
      name = "create_extra_link_time_library",
      documented = false,
      doc =
          "Creates a custom ExtraLinkTimeLibrary object. Extra keyword arguments are passed to the"
              + " provided build function when build_libraries is called. Arguments that are"
              + " depsets will be added transitively when these are combined via"
              + " cc_common.merge_cc_infos. For arguments that are not depsets, only one copy will"
              + " be maintained.",
      parameters = {
        @Param(name = "build_library_func", positional = false, named = true),
      },
      extraKeywords = @Param(name = "data"),
      useStarlarkThread = true)
  public Object createExtraLinkTimeLibrary(
      StarlarkCallable buildLibraryFunc, Dict<String, Object> dataSetsMap, StarlarkThread thread)
      throws EvalException {
    throw new UnsupportedOperationException();
  }

  @StarlarkMethod(
      name = "get_cc_semantics",
      doc = "Gets a native cc_semantics object from a language string, for creating cc actions.",
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "language",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            defaultValue = "unbound"),
      })
  public CppSemantics getCppSemanticsFromStarlark(Object languageUnchecked, StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    checkPrivateStarlarkificationAllowlist(thread);
    return getCppSemanticsFromUncheckedLanguage(languageUnchecked);
  }

  private CppSemantics getCppSemanticsFromUncheckedLanguage(Object languageUnchecked)
      throws EvalException {
    String languageString =
        convertFromNoneable(languageUnchecked, Language.CPP.getRepresentation());
    Language language = parseLanguage(languageString);
    return getSemantics(language);
  }
}
