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
import static com.google.devtools.build.lib.rules.cpp.CppHelper.asDict;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuiltinRestriction;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcStarlarkInternal.WrappedStarlarkActionFactory;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvEntry;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Flag;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagGroup;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.VariableWithValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.WithFeatureSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.Expandable;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.StringValueParser;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder.LinkActionConstruction;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcModuleApi;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.FormatMethod;
import java.util.List;
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
import net.starlark.java.eval.Structure;
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
        LtoBackendArtifacts,
        CcToolchainVariables,
        ConstraintValueInfo,
        StarlarkRuleContext,
        CcCompilationOutputs,
        CppModuleMap> {

  public abstract CppSemantics getSemantics();

  public abstract CppSemantics getSemantics(Language language);

  @Override
  public Provider getCcToolchainProvider() {
    return CcToolchainProvider.PROVIDER;
  }

  @Override
  public FeatureConfigurationForStarlark configureFeatures(
      Object ruleContextOrNone,
      Info toolchainInfo,
      Object languageObject,
      Sequence<?> requestedFeatures, // <String> expected
      Sequence<?> unsupportedFeatures, // <String> expected
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    StarlarkRuleContext ruleContext = nullIfNone(ruleContextOrNone, StarlarkRuleContext.class);

    String languageString = convertFromNoneable(languageObject, Language.CPP.getRepresentation());
    Language language = parseLanguage(languageString);
    // TODO(236152224): Remove the following when all Starlark objc configure_features have the
    // chance to migrate to using the language parameter.
    if (requestedFeatures.contains(CppRuleClasses.LANG_OBJC)) {
      language = Language.OBJC;
    }

    ImmutableSet<String> requestedFeaturesSet =
        ImmutableSet.copyOf(Sequence.cast(requestedFeatures, String.class, "requested_features"));
    ImmutableSet<String> unsupportedFeaturesSet =
        ImmutableSet.copyOf(
            Sequence.cast(unsupportedFeatures, String.class, "unsupported_features"));
    final CppConfiguration cppConfiguration;
    CcToolchainProvider toolchain =
        CcToolchainProvider.PROVIDER.wrapOrThrowEvalException(toolchainInfo);
    if (ruleContext == null) {
      throw Starlark.errorf(
          "Mandatory parameter 'ctx' of cc_common.configure_features is missing. "
              + "Please add 'ctx' as a named parameter. See "
              + "https://github.com/bazelbuild/bazel/issues/7793 for details.");
    } else {
      if (!ruleContext.getRuleContext().isLegalFragment(CppConfiguration.class)) {
        throw Starlark.errorf(
            "%s must declare '%s' as a required configuration fragment to access it.",
            ruleContext.getRuleContext().getRuleClassNameForLogging(),
            CppConfiguration.class.getSimpleName());
      }
      cppConfiguration = toolchain.getCppConfiguration();
      // buildOptions are only used when --incompatible_enable_cc_toolchain_resolution is flipped,
      // and that will only be flipped when --incompatible_require_ctx_in_configure_features is
      // flipped.
      getSemantics(language)
          .validateLayeringCheckFeatures(
              ruleContext.getRuleContext(),
              ruleContext.getAspectDescriptor(),
              toolchain,
              unsupportedFeaturesSet);
    }
    return FeatureConfigurationForStarlark.from(
        CcCommon.configureFeaturesOrThrowEvalException(
            requestedFeaturesSet, unsupportedFeaturesSet, language, toolchain, cppConfiguration));
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
  public CcToolchainVariables getCompileBuildVariables(
      Info ccToolchainInfo,
      FeatureConfigurationForStarlark featureConfiguration,
      Object sourceFile,
      Object outputFile,
      Object userCompileFlags,
      Object includeDirs,
      Object quoteIncludeDirs,
      Object systemIncludeDirs,
      Object frameworkIncludeDirs,
      Object defines,
      Object thinLtoIndex,
      Object thinLtoInputBitcodeFile,
      Object thinLtoOutputObjectFile,
      boolean usePic,
      boolean addLegacyCxxOptions,
      Object variablesExtension,
      Object stripOpts,
      Object inputFile,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    isCalledFromStarlarkCcCommon(thread);
    ImmutableList<VariablesExtension> variablesExtensions =
        asDict(variablesExtension).isEmpty()
            ? ImmutableList.of()
            : ImmutableList.of(new UserVariablesExtension(asDict(variablesExtension)));
    CcToolchainProvider ccToolchainProvider =
        CcToolchainProvider.PROVIDER.wrapOrThrowEvalException(ccToolchainInfo);
    CcToolchainVariables.Builder variables =
        CcToolchainVariables.builder(
                CompileBuildVariables.setupVariablesOrThrowEvalException(
                    featureConfiguration.getFeatureConfiguration(),
                    ccToolchainProvider,
                    convertFromNoneable(sourceFile, /* defaultValue= */ null),
                    convertFromNoneable(outputFile, /* defaultValue= */ null),
                    /* isCodeCoverageEnabled= */ false,
                    /* gcnoFile= */ null,
                    /* isUsingFission= */ false,
                    /* dwoFile= */ null,
                    /* ltoIndexingFile= */ null,
                    convertFromNoneable(thinLtoIndex, /* defaultValue= */ null),
                    convertFromNoneable(thinLtoInputBitcodeFile, /* defaultValue= */ null),
                    convertFromNoneable(thinLtoOutputObjectFile, /* defaultValue= */ null),
                    /* includes= */ ImmutableList.of(),
                    userFlagsToIterable(userCompileFlags),
                    /* cppModuleMap= */ null,
                    usePic,
                    /* fdoStamp= */ null,
                    /* dotdFile= */ null,
                    /* diagnosticsFile= */ null,
                    variablesExtensions,
                    /* additionalBuildVariables= */ ImmutableMap.of(),
                    /* directModuleMaps= */ ImmutableList.of(),
                    Depset.noneableCast(includeDirs, String.class, "framework_include_directories"),
                    Depset.noneableCast(
                        quoteIncludeDirs, String.class, "quote_include_directories"),
                    Depset.noneableCast(
                        systemIncludeDirs, String.class, "system_include_directories"),
                    Depset.noneableCast(
                        frameworkIncludeDirs, String.class, "framework_include_directories"),
                    Depset.noneableCast(defines, String.class, "preprocessor_defines").toList(),
                    ImmutableList.of()))
            .addStringSequenceVariable("stripopts", asClassImmutableList(stripOpts));
    String inputFileString = convertFromNoneable(inputFile, null);
    if (inputFileString != null) {
      variables.addVariable("input_file", inputFileString);
    }
    return variables.build();
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

  /** Converts an object that can be either Sequence, or None into ImmutableList. */
  protected ImmutableList<String> asStringImmutableList(Object o) {
    Sequence<String> starlarkList =
        convertFromNoneable(o, /* defaultValue= */ (Sequence<String>) null);
    if (starlarkList != null) {
      return starlarkList.getImmutableList();
    } else {
      return ImmutableList.of();
    }
  }

  /** Converts an object that represents user flags as either Sequence or None into Iterable. */
  protected Iterable<String> userFlagsToIterable(Object o) throws EvalException {
    if (o instanceof Sequence) {
      return asStringImmutableList(o);
    } else if (o instanceof NoneType) {
      return ImmutableList.of();
    } else {
      throw Starlark.errorf("Only list is allowed.");
    }
  }

  @SuppressWarnings("unchecked")
  @Nullable
  protected <T> ImmutableList<T> asClassImmutableList(Object o) {
    if (o == Starlark.UNBOUND) {
      return ImmutableList.of();
    } else {
      ImmutableList<T> list = ((Sequence<T>) o).getImmutableList();
      if (list.isEmpty()) {
        return ImmutableList.of();
      }
      return list;
    }
  }

  @Nullable
  protected <T> Object asClassImmutableListOrNestedSet(
      Object o, Class<T> tClass, String description) throws EvalException {
    if (o == Starlark.UNBOUND) {
      return ImmutableList.of();
    } else {
      return o instanceof Depset
          ? Depset.cast(o, tClass, description)
          : Sequence.cast(o, tClass, description).getImmutableList();
    }
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
      Object unused3,
      Object moduleMap,
      Object unused1,
      Object unused2,
      Object externalIncludes,
      Object virtualToOriginalHeaders,
      Sequence<?> dependentCcCompilationContexts,
      Sequence<?> exportedDependentCcCompilationContexts,
      Sequence<?> nonCodeInputs,
      Sequence<?> looseHdrsDirsObject,
      String headersCheckingMode,
      Boolean propagateModuleMapToCompileAction,
      Object picHeaderModule,
      Object headerModule,
      Sequence<?> separateModuleHeaders,
      Object separateModule,
      Object separatePicModule,
      Object addPublicHeadersToModularHeaders,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);

    CcCompilationContext.Builder ccCompilationContext = CcCompilationContext.builder();

    // Public parameters.
    ImmutableList<Artifact> headerList = toNestedSetOfArtifacts(headers, "headers").toList();
    ccCompilationContext.addDeclaredIncludeSrcs(headerList);
    ImmutableList<Artifact> textualHdrsList =
        Sequence.cast(directTextualHdrs, Artifact.class, "direct_textual_headers")
            .getImmutableList();
    ImmutableList<Artifact> modularPublicHdrsList =
        Sequence.cast(directPublicHdrs, Artifact.class, "direct_public_headers").getImmutableList();
    ImmutableList<Artifact> modularPrivateHdrsList =
        Sequence.cast(directPrivateHdrs, Artifact.class, "direct_private_headers")
            .getImmutableList();

    ccCompilationContext.addSystemIncludeDirs(
        toNestedSetOfStrings(systemIncludes, "system_includes").toList().stream()
            .map(x -> PathFragment.create(x))
            .collect(toImmutableList()));
    ccCompilationContext.addIncludeDirs(
        toNestedSetOfStrings(includes, "includes").toList().stream()
            .map(x -> PathFragment.create(x))
            .collect(toImmutableList()));
    ccCompilationContext.addQuoteIncludeDirs(
        toNestedSetOfStrings(quoteIncludes, "quote_includes").toList().stream()
            .map(x -> PathFragment.create(x))
            .collect(toImmutableList()));
    ccCompilationContext.addFrameworkIncludeDirs(
        toNestedSetOfStrings(frameworkIncludes, "framework_includes").toList().stream()
            .map(x -> PathFragment.create(x))
            .collect(toImmutableList()));
    ccCompilationContext.addDefines(toNestedSetOfStrings(defines, "defines").toList());
    ccCompilationContext.addNonTransitiveDefines(
        toNestedSetOfStrings(localDefines, "local_defines").toList());
    ccCompilationContext.addTextualHdrs(textualHdrsList);
    ccCompilationContext.addModularPublicHdrs(modularPublicHdrsList);
    ccCompilationContext.addModularPrivateHdrs(modularPrivateHdrsList);

    if (moduleMap != null && moduleMap != Starlark.UNBOUND && moduleMap != Starlark.NONE) {
      ccCompilationContext.setCppModuleMap((CppModuleMap) moduleMap);
    }

    ccCompilationContext.addExternalIncludeDirs(
        toNestedSetOfStrings(externalIncludes, "external_includes").toList().stream()
            .map(PathFragment::create)
            .collect(toImmutableList()));

    ccCompilationContext.addVirtualToOriginalHeaders(
        Depset.cast(virtualToOriginalHeaders, Tuple.class, "virtual_to_original_headers"));

    ccCompilationContext.addDependentCcCompilationContexts(
        Sequence.cast(
                exportedDependentCcCompilationContexts,
                CcCompilationContext.class,
                "exported_dependent_cc_compilation_contexts")
            .getImmutableList(),
        Sequence.cast(
                dependentCcCompilationContexts,
                CcCompilationContext.class,
                "dependent_cc_compilation_contexts")
            .getImmutableList());

    ccCompilationContext.addNonCodeInputs(
        Sequence.cast(nonCodeInputs, Artifact.class, "non_code_inputs").getImmutableList());

    ccCompilationContext.setPropagateCppModuleMapAsActionInput(propagateModuleMapToCompileAction);
    ccCompilationContext.setPicHeaderModule(
        picHeaderModule == Starlark.NONE ? null : (Artifact.DerivedArtifact) picHeaderModule);
    ccCompilationContext.setHeaderModule(
        headerModule == Starlark.NONE ? null : (Artifact.DerivedArtifact) headerModule);
    ccCompilationContext.setSeparateModuleHdrs(
        Sequence.cast(separateModuleHeaders, Artifact.class, "separate_module_headers"),
        convertFromNoneable(separateModule, null),
        convertFromNoneable(separatePicModule, null));

    if ((Boolean) addPublicHeadersToModularHeaders) {
      ccCompilationContext.addModularPublicHdrs(headerList);
    }

    return ccCompilationContext.build();
  }

  @Override
  public CcCompilationOutputs mergeCcCompilationOutputsFromStarlark(
      Sequence<?> compilationOutputs, // <CcCompilationOutputs>
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    CcCompilationOutputs.Builder ccCompilationOutputsBuilder = CcCompilationOutputs.builder();
    for (CcCompilationOutputs ccCompilationOutputs :
        Sequence.cast(compilationOutputs, CcCompilationOutputs.class, "compilation_outputs")) {
      ccCompilationOutputsBuilder.merge(ccCompilationOutputs);
    }
    return ccCompilationOutputsBuilder.build();
  }

  @Override
  public CcCompilationContext mergeCompilationContexts(
      Sequence<?> compilationContexts,
      Sequence<?> nonExportedCompilationContexts,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    if (compilationContexts.isEmpty() && nonExportedCompilationContexts.isEmpty()) {
      return CcCompilationContext.EMPTY;
    }
    return CcCompilationContext.builder()
        .addDependentCcCompilationContexts(
            Sequence.cast(compilationContexts, CcCompilationContext.class, "compilation_contexts"),
            Sequence.cast(
                nonExportedCompilationContexts,
                CcCompilationContext.class,
                "non_exported_compilation_contexts"))
        .build();
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

  @Override
  public CppModuleMap createCppModuleMap(Artifact file, String name, StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    return new CppModuleMap(file, name);
  }

  /**
   * Create an LTO backend, using the appropriate constructor depending on whether the associated
   * ThinLTO link will utilize LTO indexing (therefore unique LTO backend actions), or not (and
   * therefore the library being linked will create a set of shared LTO backends).
   *
   * <p>TODO(b/128341904): Do cross module optimization once there is Starlark support.
   */
  @Override
  public LtoBackendArtifacts createLtoBackendArtifacts(
      Object starlarkRuleContextObj,
      Object actionsObj,
      String ltoOutputRootPrefixString,
      String ltoObjRootPrefixString,
      Artifact bitcodeFile,
      Object allBitcodeFilesObj,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      Info ccToolchainInfo,
      StructImpl fdoContextStruct,
      boolean usePic,
      boolean shouldCreatePerObjectDebugInfo,
      boolean createSharedNonLto,
      Sequence<?> argv,
      StarlarkThread thread)
      throws EvalException, InterruptedException, RuleErrorException {
    isCalledFromStarlarkCcCommon(thread);
    LinkActionConstruction actionConstruction;
    // TODO(b/331164666): remove uses of `ctx`, cleanup uses of newActionConstruction
    if (actionsObj instanceof StarlarkActionFactory actions) {
      if (actions instanceof WrappedStarlarkActionFactory wrapped) {
        actionConstruction = wrapped.construction;
      } else {
        actionConstruction = CppLinkActionBuilder.newActionConstruction(actions.getRuleContext());
      }
    } else if (starlarkRuleContextObj instanceof StarlarkRuleContext starlarkRuleContext) {
      actionConstruction =
          CppLinkActionBuilder.newActionConstruction(starlarkRuleContext.getRuleContext());
    } else {
      throw Starlark.errorf("'actions' parameter is mandatory ('ctx' deprecated).");
    }
    // Depending on whether LTO indexing is allowed, generate an LTO backend
    // that will be fed the results of the indexing step, or a dummy LTO backend
    // that simply compiles the bitcode into native code without any index-based
    // cross module optimization.
    if (createSharedNonLto) {
      actionConstruction =
          new LinkActionConstruction(
              actionConstruction.getContext(),
              actionConstruction.getConfig(),
              /* shareableArtifacts= */ true);
    }
    PathFragment ltoOutputRootPrefix = PathFragment.create(ltoOutputRootPrefixString);
    PathFragment ltoObjRootPrefix = PathFragment.create(ltoObjRootPrefixString);
    CcToolchainProvider ccToolchain =
        CcToolchainProvider.PROVIDER.wrapOrThrowEvalException(ccToolchainInfo);
    LtoBackendArtifacts ltoBackendArtifacts;
    ltoBackendArtifacts =
        new LtoBackendArtifacts(
            ltoOutputRootPrefix,
            ltoObjRootPrefix,
            bitcodeFile,
            allBitcodeFilesObj == Starlark.NONE
                ? null
                : Depset.noneableCast(allBitcodeFilesObj, Artifact.class, "all_bitcode_files"),
            actionConstruction,
            featureConfigurationForStarlark.getFeatureConfiguration(),
            ccToolchain,
            new FdoContext(fdoContextStruct),
            usePic,
            shouldCreatePerObjectDebugInfo,
            Sequence.cast(argv, String.class, "argv"));
    return ltoBackendArtifacts;
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
  public boolean addGoExecGroupsToBinaryRules(StarlarkThread thread) throws EvalException {
    // This method is called from cc_common.bzl and semantics.bzl
    if (!isStarlarkCcCommonCalledFromBuiltins(thread)) {
      throw Starlark.errorf("add_go_exec_groups_to_binary_rules can only be used in builtins");
    }
    return thread.getSemantics().getBool(BuildLanguageOptions.ADD_GO_EXEC_GROUPS_TO_BINARY_RULES);
  }



  // TODO(b/65151735): Remove when cc_flags is entirely from features.
  @Override
  public String legacyCcFlagsMakeVariable(Info ccToolchainInfo, StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    CcToolchainProvider ccToolchain =
        CcToolchainProvider.PROVIDER.wrapOrThrowEvalException(ccToolchainInfo);
    return ccToolchain.getLegacyCcFlagsMakeVariable();
  }

  /** Converts None, or a Sequence, or a Depset to a NestedSet. */
  private static <T> NestedSet<T> convertToNestedSet(Object o, Class<T> type, String fieldName)
      throws EvalException {
    if (o == Starlark.UNBOUND || o == Starlark.NONE) {
      return NestedSetBuilder.emptySet(Order.COMPILE_ORDER);
    }
    return o instanceof Depset
        ? Depset.cast(o, type, fieldName)
        : NestedSetBuilder.wrap(Order.COMPILE_ORDER, Sequence.cast(o, type, fieldName));
  }

  @FormatMethod
  private static EvalException infoError(Info info, String format, Object... args) {
    return Starlark.errorf(
        "in %s instantiated at %s: %s",
        info.getProvider().getPrintableName(),
        info.getCreationLocation(),
        String.format(format, args));
  }

  /** Checks whether the {@link StarlarkInfo} is of the required type. */
  private static void checkRightProviderType(StarlarkInfo provider, String type)
      throws EvalException {
    String providerType = (String) getValueOrNull(provider, "type_name");
    if (providerType == null) {
      providerType = provider.getProvider().getPrintableName();
    }
    if (!type.equals(provider.getValue("type_name"))) {
      throw infoError(provider, "Expected object of type '%s', received '%s'.", type, providerType);
    }
  }

  @Nullable
  private static Object getValueOrNull(Structure x, String name) {
    try {
      return x.getValue(name);
    } catch (EvalException e) {
      return null;
    }
  }

  /** Creates a {@link Feature} from a {@link StarlarkInfo}. */
  @VisibleForTesting
  static Feature featureFromStarlark(StarlarkInfo featureStruct) throws EvalException {
    checkRightProviderType(featureStruct, "feature");
    String name = getMandatoryFieldFromStarlarkProvider(featureStruct, "name", String.class);
    Boolean enabled =
        getMandatoryFieldFromStarlarkProvider(featureStruct, "enabled", Boolean.class);
    if (name == null || (name.isEmpty() && !enabled)) {
      throw infoError(
          featureStruct, "A feature must either have a nonempty 'name' field or be enabled.");
    }

    if (!name.matches("^[_a-z0-9+\\-\\.]*$")) {
      throw infoError(
          featureStruct,
          "A feature's name must consist solely of lowercase ASCII letters, digits, '.', "
              + "'_', '+', and '-', got '%s'",
          name);
    }

    ImmutableList.Builder<FlagSet> flagSetBuilder = ImmutableList.builder();
    ImmutableList<StarlarkInfo> flagSets =
        getStarlarkProviderListFromStarlarkField(featureStruct, "flag_sets");
    for (StarlarkInfo flagSetObject : flagSets) {
      FlagSet flagSet = flagSetFromStarlark(flagSetObject, /* actionName= */ null);
      if (flagSet.getActions().isEmpty()) {
        throw infoError(
            flagSetObject,
            "A flag_set that belongs to a feature must have nonempty 'actions' parameter.");
      }
      flagSetBuilder.add(flagSet);
    }

    ImmutableList.Builder<EnvSet> envSetBuilder = ImmutableList.builder();
    ImmutableList<StarlarkInfo> envSets =
        getStarlarkProviderListFromStarlarkField(featureStruct, "env_sets");
    for (StarlarkInfo envSet : envSets) {
      envSetBuilder.add(envSetFromStarlark(envSet));
    }

    ImmutableList.Builder<ImmutableSet<String>> requiresBuilder = ImmutableList.builder();

    ImmutableList<StarlarkInfo> requires =
        getStarlarkProviderListFromStarlarkField(featureStruct, "requires");
    for (StarlarkInfo featureSetStruct : requires) {
      if (!"feature_set".equals(featureSetStruct.getValue("type_name"))) { // getValue() may be null
        throw infoError(featureStruct, "expected object of type 'feature_set'.");
      }
      ImmutableSet<String> featureSet =
          getStringSetFromStarlarkProviderField(featureSetStruct, "features");
      requiresBuilder.add(featureSet);
    }

    ImmutableList<String> implies =
        getStringListFromStarlarkProviderField(featureStruct, "implies");

    ImmutableList<String> provides =
        getStringListFromStarlarkProviderField(featureStruct, "provides");

    return new Feature(
        name,
        flagSetBuilder.build(),
        envSetBuilder.build(),
        enabled,
        requiresBuilder.build(),
        implies,
        provides);
  }

  /**
   * Creates a Pair(name, value) that represents a {@link
   * com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.MakeVariable} from a {@link
   * StarlarkInfo}.
   */
  @VisibleForTesting
  static Pair<String, String> makeVariableFromStarlark(StarlarkInfo makeVariableStruct)
      throws EvalException {
    checkRightProviderType(makeVariableStruct, "make_variable");
    String name = getMandatoryFieldFromStarlarkProvider(makeVariableStruct, "name", String.class);
    String value = getMandatoryFieldFromStarlarkProvider(makeVariableStruct, "value", String.class);
    if (name == null || name.isEmpty()) {
      throw infoError(
          makeVariableStruct, "'name' parameter of make_variable must be a nonempty string.");
    }
    if (value == null || value.isEmpty()) {
      throw infoError(
          makeVariableStruct, "'value' parameter of make_variable must be a nonempty string.");
    }
    return Pair.of(name, value);
  }

  /**
   * Creates a Pair(name, path) that represents a {@link
   * com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath} from a {@link
   * StarlarkInfo}.
   */
  @VisibleForTesting
  static Pair<String, String> toolPathFromStarlark(StarlarkInfo toolPathStruct)
      throws EvalException {
    checkRightProviderType(toolPathStruct, "tool_path");
    String name = getMandatoryFieldFromStarlarkProvider(toolPathStruct, "name", String.class);
    String path = getMandatoryFieldFromStarlarkProvider(toolPathStruct, "path", String.class);
    if (name == null || name.isEmpty()) {
      throw infoError(toolPathStruct, "'name' parameter of tool_path must be a nonempty string.");
    }
    if (path == null || path.isEmpty()) {
      throw infoError(toolPathStruct, "'path' parameter of tool_path must be a nonempty string.");
    }
    return Pair.of(name, path);
  }

  /** Creates a {@link VariableWithValue} from a {@link StarlarkInfo}. */
  @VisibleForTesting
  static VariableWithValue variableWithValueFromStarlark(StarlarkInfo variableWithValueStruct)
      throws EvalException {
    checkRightProviderType(variableWithValueStruct, "variable_with_value");
    String name =
        getMandatoryFieldFromStarlarkProvider(variableWithValueStruct, "name", String.class);
    String value =
        getMandatoryFieldFromStarlarkProvider(variableWithValueStruct, "value", String.class);
    if (name == null || name.isEmpty()) {
      throw infoError(
          variableWithValueStruct,
          "'name' parameter of variable_with_value must be a nonempty string.");
    }
    if (value == null || value.isEmpty()) {
      throw infoError(
          variableWithValueStruct,
          "'value' parameter of variable_with_value must be a nonempty string.");
    }
    return new VariableWithValue(name, value);
  }

  /** Creates an {@link EnvEntry} from a {@link StarlarkInfo}. */
  @VisibleForTesting
  static EnvEntry envEntryFromStarlark(StarlarkInfo envEntryStruct) throws EvalException {
    checkRightProviderType(envEntryStruct, "env_entry");
    String key = getMandatoryFieldFromStarlarkProvider(envEntryStruct, "key", String.class);
    String value = getMandatoryFieldFromStarlarkProvider(envEntryStruct, "value", String.class);
    if (key == null || key.isEmpty()) {
      throw infoError(envEntryStruct, "'key' parameter of env_entry must be a nonempty string.");
    }
    if (value == null || value.isEmpty()) {
      throw infoError(envEntryStruct, "'value' parameter of env_entry must be a nonempty string.");
    }
    String expandIfAvailable =
        getOptionalFieldFromStarlarkProvider(envEntryStruct, "expand_if_available", String.class);
    StringValueParser parser = new StringValueParser(value);
    return new EnvEntry(
        key,
        parser.getChunks(),
        expandIfAvailable == null ? ImmutableSet.of() : ImmutableSet.of(expandIfAvailable));
  }

  /** Creates a {@link WithFeatureSet} from a {@link StarlarkInfo}. */
  @VisibleForTesting
  static WithFeatureSet withFeatureSetFromStarlark(StarlarkInfo withFeatureSetStruct)
      throws EvalException {
    checkRightProviderType(withFeatureSetStruct, "with_feature_set");
    ImmutableSet<String> features =
        getStringSetFromStarlarkProviderField(withFeatureSetStruct, "features");
    ImmutableSet<String> notFeatures =
        getStringSetFromStarlarkProviderField(withFeatureSetStruct, "not_features");
    return new WithFeatureSet(features, notFeatures);
  }

  /** Creates an {@link EnvSet} from a {@link StarlarkInfo}. */
  @VisibleForTesting
  static EnvSet envSetFromStarlark(StarlarkInfo envSetStruct) throws EvalException {
    checkRightProviderType(envSetStruct, "env_set");
    ImmutableSet<String> actions = getStringSetFromStarlarkProviderField(envSetStruct, "actions");
    if (actions.isEmpty()) {
      throw infoError(envSetStruct, "actions parameter of env_set must be a nonempty list.");
    }
    ImmutableList.Builder<EnvEntry> envEntryBuilder = ImmutableList.builder();
    ImmutableList<StarlarkInfo> envEntryStructs =
        getStarlarkProviderListFromStarlarkField(envSetStruct, "env_entries");
    for (StarlarkInfo envEntryStruct : envEntryStructs) {
      envEntryBuilder.add(envEntryFromStarlark(envEntryStruct));
    }

    ImmutableSet.Builder<WithFeatureSet> withFeatureSetBuilder = ImmutableSet.builder();
    ImmutableList<StarlarkInfo> withFeatureSetStructs =
        getStarlarkProviderListFromStarlarkField(envSetStruct, "with_features");
    for (StarlarkInfo withFeatureSetStruct : withFeatureSetStructs) {
      withFeatureSetBuilder.add(withFeatureSetFromStarlark(withFeatureSetStruct));
    }
    return new EnvSet(actions, envEntryBuilder.build(), withFeatureSetBuilder.build());
  }

  /** Creates a {@link FlagGroup} from a {@link StarlarkInfo}. */
  @VisibleForTesting
  static FlagGroup flagGroupFromStarlark(StarlarkInfo flagGroupStruct) throws EvalException {
    checkRightProviderType(flagGroupStruct, "flag_group");

    ImmutableList.Builder<Expandable> expandableBuilder = ImmutableList.builder();
    ImmutableList<String> flags = getStringListFromStarlarkProviderField(flagGroupStruct, "flags");
    for (String flag : flags) {
      StringValueParser parser = new StringValueParser(flag);
      expandableBuilder.add(Flag.create(parser.getChunks()));
    }

    ImmutableList<StarlarkInfo> flagGroups =
        getStarlarkProviderListFromStarlarkField(flagGroupStruct, "flag_groups");
    for (StarlarkInfo flagGroup : flagGroups) {
      expandableBuilder.add(flagGroupFromStarlark(flagGroup));
    }

    if (flagGroups.size() > 0 && flags.size() > 0) {
      throw infoError(
          flagGroupStruct,
          "flag_group must contain either a list of flags or a list of flag_groups.");
    }

    if (flagGroups.size() == 0 && flags.size() == 0) {
      throw infoError(flagGroupStruct, "Both 'flags' and 'flag_groups' are empty.");
    }

    String iterateOver =
        getMandatoryFieldFromStarlarkProvider(flagGroupStruct, "iterate_over", String.class);
    String expandIfAvailable =
        getMandatoryFieldFromStarlarkProvider(flagGroupStruct, "expand_if_available", String.class);
    String expandIfNotAvailable =
        getMandatoryFieldFromStarlarkProvider(
            flagGroupStruct, "expand_if_not_available", String.class);
    String expandIfTrue =
        getMandatoryFieldFromStarlarkProvider(flagGroupStruct, "expand_if_true", String.class);
    String expandIfFalse =
        getMandatoryFieldFromStarlarkProvider(flagGroupStruct, "expand_if_false", String.class);
    StarlarkInfo expandIfEqualStruct =
        getMandatoryFieldFromStarlarkProvider(
            flagGroupStruct, "expand_if_equal", StarlarkInfo.class);
    VariableWithValue expandIfEqual =
        expandIfEqualStruct == null ? null : variableWithValueFromStarlark(expandIfEqualStruct);

    return new FlagGroup(
        expandableBuilder.build(),
        iterateOver,
        expandIfAvailable == null ? ImmutableSet.of() : ImmutableSet.of(expandIfAvailable),
        expandIfNotAvailable == null ? ImmutableSet.of() : ImmutableSet.of(expandIfNotAvailable),
        expandIfTrue,
        expandIfFalse,
        expandIfEqual);
  }

  /** Creates a {@link FlagSet} from a {@link StarlarkInfo}. */
  @VisibleForTesting
  static FlagSet flagSetFromStarlark(StarlarkInfo flagSetStruct, String actionName)
      throws EvalException {
    checkRightProviderType(flagSetStruct, "flag_set");
    ImmutableSet<String> actions = getStringSetFromStarlarkProviderField(flagSetStruct, "actions");
    // if we are creating a flag set for an action_config, we need to propagate the name of the
    // action to its flag_set.action_names
    if (actionName != null) {
      if (!actions.isEmpty()) {
        throw Starlark.errorf(ActionConfig.FLAG_SET_WITH_ACTION_ERROR, actionName);
      }
      actions = ImmutableSet.of(actionName);
    }
    ImmutableList.Builder<FlagGroup> flagGroupsBuilder = ImmutableList.builder();
    ImmutableList<StarlarkInfo> flagGroups =
        getStarlarkProviderListFromStarlarkField(flagSetStruct, "flag_groups");
    for (StarlarkInfo flagGroup : flagGroups) {
      flagGroupsBuilder.add(flagGroupFromStarlark(flagGroup));
    }

    ImmutableSet.Builder<WithFeatureSet> withFeatureSetBuilder = ImmutableSet.builder();
    ImmutableList<StarlarkInfo> withFeatureSetStructs =
        getStarlarkProviderListFromStarlarkField(flagSetStruct, "with_features");
    for (StarlarkInfo withFeatureSetStruct : withFeatureSetStructs) {
      withFeatureSetBuilder.add(withFeatureSetFromStarlark(withFeatureSetStruct));
    }

    return new FlagSet(
        actions, ImmutableSet.of(), withFeatureSetBuilder.build(), flagGroupsBuilder.build());
  }

  /**
   * Creates a {@link com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Tool} from a
   * {@link StarlarkInfo}.
   */
  @VisibleForTesting
  static CcToolchainFeatures.Tool toolFromStarlark(StarlarkInfo toolStruct, OS execOs)
      throws EvalException {
    checkRightProviderType(toolStruct, "tool");

    String toolPathString = getOptionalFieldFromStarlarkProvider(toolStruct, "path", String.class);
    Artifact toolArtifact =
        getOptionalFieldFromStarlarkProvider(toolStruct, "tool", Artifact.class);

    PathFragment toolPath;
    CcToolchainFeatures.Tool.PathOrigin toolPathOrigin;
    if (toolPathString != null) {
      if (toolArtifact != null) {
        throw infoError(toolStruct, "\"tool\" and \"path\" cannot be set at the same time.");
      }

      toolPath = PathFragment.createForOs(toolPathString, execOs);
      if (toolPath.isEmpty()) {
        throw infoError(toolStruct, "The 'path' field of tool must be a nonempty string.");
      }

      if (toolPath.isAbsolute()) {
        toolPathOrigin = CcToolchainFeatures.Tool.PathOrigin.FILESYSTEM_ROOT;
      } else {
        toolPathOrigin = CcToolchainFeatures.Tool.PathOrigin.CROSSTOOL_PACKAGE;
      }
    } else if (toolArtifact != null) {
      toolPath = toolArtifact.getExecPath();
      toolPathOrigin = CcToolchainFeatures.Tool.PathOrigin.WORKSPACE_ROOT;
    } else {
      throw Starlark.errorf("Exactly one of \"tool\" and \"path\" must be set.");
    }
    Preconditions.checkState(toolPath != null && toolPathOrigin != null);

    ImmutableSet.Builder<WithFeatureSet> withFeatureSetBuilder = ImmutableSet.builder();
    ImmutableList<StarlarkInfo> withFeatureSetStructs =
        getStarlarkProviderListFromStarlarkField(toolStruct, "with_features");
    for (StarlarkInfo withFeatureSetStruct : withFeatureSetStructs) {
      withFeatureSetBuilder.add(withFeatureSetFromStarlark(withFeatureSetStruct));
    }

    ImmutableSet<String> executionRequirements =
        getStringSetFromStarlarkProviderField(toolStruct, "execution_requirements");
    return new CcToolchainFeatures.Tool(
        toolPath, toolPathOrigin, executionRequirements, withFeatureSetBuilder.build());
  }

  /** Creates an {@link ActionConfig} from a {@link StarlarkInfo}. */
  @VisibleForTesting
  static ActionConfig actionConfigFromStarlark(StarlarkInfo actionConfigStruct, OS execOs)
      throws EvalException {
    checkRightProviderType(actionConfigStruct, "action_config");
    String actionName =
        getMandatoryFieldFromStarlarkProvider(actionConfigStruct, "action_name", String.class);
    if (actionName == null || actionName.isEmpty()) {
      throw infoError(
          actionConfigStruct,
          "The 'action_name' field of action_config must be a nonempty string.");
    }
    if (!actionName.matches("^[_a-z0-9+\\-\\.]*$")) {
      throw infoError(
          actionConfigStruct,
          "An action_config's name must consist solely of lowercase ASCII letters, digits, "
              + "'.', '_', '+', and '-', got '%s'",
          actionName);
    }

    Boolean enabled =
        getMandatoryFieldFromStarlarkProvider(actionConfigStruct, "enabled", Boolean.class);

    ImmutableList.Builder<CcToolchainFeatures.Tool> toolBuilder = ImmutableList.builder();
    ImmutableList<StarlarkInfo> toolStructs =
        getStarlarkProviderListFromStarlarkField(actionConfigStruct, "tools");
    for (StarlarkInfo toolStruct : toolStructs) {
      toolBuilder.add(toolFromStarlark(toolStruct, execOs));
    }

    ImmutableList.Builder<FlagSet> flagSetBuilder = ImmutableList.builder();
    ImmutableList<StarlarkInfo> flagSets =
        getStarlarkProviderListFromStarlarkField(actionConfigStruct, "flag_sets");
    for (StarlarkInfo flagSet : flagSets) {
      flagSetBuilder.add(flagSetFromStarlark(flagSet, actionName));
    }

    ImmutableList<String> implies =
        getStringListFromStarlarkProviderField(actionConfigStruct, "implies");

    return new ActionConfig(
        actionName, actionName, toolBuilder.build(), flagSetBuilder.build(), enabled, implies);
  }

  @VisibleForTesting
  interface ArtifactNamePatternAdder {
    void add(ArtifactCategory category, String prefix, String extension);
  }

  @VisibleForTesting
  static void artifactNamePatternFromStarlark(
      StarlarkInfo artifactNamePatternStruct, ArtifactNamePatternAdder adder) throws EvalException {
    checkRightProviderType(artifactNamePatternStruct, "artifact_name_pattern");
    String categoryName =
        getMandatoryFieldFromStarlarkProvider(
            artifactNamePatternStruct, "category_name", String.class);
    if (categoryName == null || categoryName.isEmpty()) {
      throw infoError(
          artifactNamePatternStruct,
          "The 'category_name' field of artifact_name_pattern must be a nonempty string.");
    }
    ArtifactCategory foundCategory = null;
    for (ArtifactCategory artifactCategory : ArtifactCategory.values()) {
      if (categoryName.equals(artifactCategory.getCategoryName())) {
        foundCategory = artifactCategory;
      }
    }

    if (foundCategory == null) {
      throw infoError(
          artifactNamePatternStruct, "Artifact category %s not recognized.", categoryName);
    }

    String extension =
        Strings.nullToEmpty(
            getMandatoryFieldFromStarlarkProvider(
                artifactNamePatternStruct, "extension", String.class));
    if (!foundCategory.getAllowedExtensions().contains(extension)) {
      throw infoError(
          artifactNamePatternStruct,
          "Unrecognized file extension '%s', allowed extensions are %s,"
              + " please check artifact_name_pattern configuration for %s in your rule.",
          extension,
          StringUtil.joinEnglishListSingleQuoted(foundCategory.getAllowedExtensions()),
          foundCategory.getCategoryName());
    }

    String prefix =
        Strings.nullToEmpty(
            getMandatoryFieldFromStarlarkProvider(
                artifactNamePatternStruct, "prefix", String.class));
    adder.add(foundCategory, prefix, extension);
  }

  private static <T> T getOptionalFieldFromStarlarkProvider(
      StarlarkInfo provider, String fieldName, Class<T> clazz) throws EvalException {
    return getFieldFromStarlarkProvider(provider, fieldName, clazz, false);
  }

  private static <T> T getMandatoryFieldFromStarlarkProvider(
      StarlarkInfo provider, String fieldName, Class<T> clazz) throws EvalException {
    return getFieldFromStarlarkProvider(provider, fieldName, clazz, true);
  }

  private static <T> T getFieldFromStarlarkProvider(
      StarlarkInfo provider, String fieldName, Class<T> clazz, boolean mandatory)
      throws EvalException {
    Object obj = provider.getValue(fieldName);
    if (obj == null) {
      if (mandatory) {
        throw infoError(provider, "Missing mandatory field '%s'", fieldName);
      }
      return null;
    }
    if (clazz.isInstance(obj)) {
      return clazz.cast(obj);
    }
    if (NoneType.class.isInstance(obj)) {
      return null;
    }
    throw infoError(provider, "Field '%s' is not of '%s' type.", fieldName, clazz.getName());
  }

  /** Returns a list of strings from a field of a {@link StarlarkInfo}. */
  private static ImmutableList<String> getStringListFromStarlarkProviderField(
      StarlarkInfo provider, String fieldName) throws EvalException {
    Object v = getValueOrNull(provider, fieldName);
    return v == null
        ? ImmutableList.of()
        : ImmutableList.copyOf(Sequence.noneableCast(v, String.class, fieldName));
  }

  /** Returns a set of strings from a field of a {@link StarlarkInfo}. */
  private static ImmutableSet<String> getStringSetFromStarlarkProviderField(
      StarlarkInfo provider, String fieldName) throws EvalException {
    Object v = getValueOrNull(provider, fieldName);
    return v == null
        ? ImmutableSet.of()
        : ImmutableSet.copyOf(Sequence.noneableCast(v, String.class, fieldName));
  }

  /** Returns a list of StarlarkInfo providers from a field of a {@link StarlarkInfo}. */
  private static ImmutableList<StarlarkInfo> getStarlarkProviderListFromStarlarkField(
      StarlarkInfo provider, String fieldName) throws EvalException {
    Object v = getValueOrNull(provider, fieldName);
    return v == null
        ? ImmutableList.of()
        : ImmutableList.copyOf(Sequence.noneableCast(v, StarlarkInfo.class, fieldName));
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
        && !label.getCanonicalForm().contains("_builtins//:common/cc/link")) {
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

  protected Label getCallerLabel(StarlarkActionFactory actions, String name) throws EvalException {
    try {
      return Label.create(
          actions.getRuleContext().getActionOwner().getLabel().getPackageIdentifier(), name);
    } catch (LabelSyntaxException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
  }

  @Override
  @SuppressWarnings("unchecked")
  public LtoCompilationContext createLtoCompilationContextFromStarlark(
      Object objectsObject, StarlarkThread thread) throws EvalException {
    checkPrivateStarlarkificationAllowlist(thread);
    Dict<Artifact, Tuple> objects =
        Dict.cast(objectsObject, Artifact.class, Tuple.class, "objects");
    LtoCompilationContext.Builder builder = new LtoCompilationContext.Builder();
    for (Artifact k : objects) {
      Tuple t = objects.get(k);
      if (t.size() != 2) {
        throw new EvalException(
            "wrong length tuple for an (index_file, copts), want 2, got " + t.size());
      }
      Object minimizedBitcode = t.get(0);
      if (!(minimizedBitcode instanceof Artifact)) {
        throw new EvalException("expected Artifact for minimized bitcode, got something else");
      }
      Object copts = t.get(1);
      if (!(copts instanceof StarlarkList)) {
        throw new EvalException("expected list for copts, got something else");
      }
      builder.addBitcodeFile(
          k, (Artifact) minimizedBitcode, ImmutableList.copyOf((StarlarkList<String>) copts));
    }
    return builder.build();
  }

  @Override
  public CcCompilationOutputs createCompilationOutputsFromStarlark(
      Object objectsObject,
      Object picObjectsObject,
      Object ltoCompilationContextObject,
      Object dwoObjectsObject,
      Object picDwoObjectsObject,
      StarlarkThread thread)
      throws EvalException {
    isCalledFromStarlarkCcCommon(thread);
    CcCompilationOutputs.Builder ccCompilationOutputsBuilder = CcCompilationOutputs.builder();
    NestedSet<Artifact> objects = convertToNestedSet(objectsObject, Artifact.class, "objects");
    validateExtensions(
        "objects",
        objects.toList(),
        Link.OBJECT_FILETYPES,
        Link.OBJECT_FILETYPES,
        /* allowAnyTreeArtifacts= */ true);
    LtoCompilationContext ltoCompilationContext =
        convertFromNoneable(ltoCompilationContextObject, null);
    NestedSet<Artifact> picObjects =
        convertToNestedSet(picObjectsObject, Artifact.class, "pic_objects");
    validateExtensions(
        "pic_objects",
        picObjects.toList(),
        Link.OBJECT_FILETYPES,
        Link.OBJECT_FILETYPES,
        /* allowAnyTreeArtifacts= */ true);
    ccCompilationOutputsBuilder.addObjectFiles(objects.toList());
    ccCompilationOutputsBuilder.addPicObjectFiles(picObjects.toList());
    if (ltoCompilationContext != null) {
      ccCompilationOutputsBuilder.addLtoCompilationContext(ltoCompilationContext);
    }
    NestedSet<Artifact> dwoObjects =
        convertToNestedSet(dwoObjectsObject, Artifact.class, "dwo_objects");
    for (Artifact dwoFile : dwoObjects.toList()) {
      ccCompilationOutputsBuilder.addDwoFile(dwoFile);
    }
    NestedSet<Artifact> picDwoObjects =
        convertToNestedSet(picDwoObjectsObject, Artifact.class, "pic_dwo_objects");
    for (Artifact picDwoFile : picDwoObjects.toList()) {
      ccCompilationOutputsBuilder.addPicDwoFile(picDwoFile);
    }
    return ccCompilationOutputsBuilder.build();
  }

  private static void validateExtensions(
      String paramName,
      List<Artifact> files,
      FileTypeSet validFileTypeSet,
      FileTypeSet fileTypeForErrorMessage,
      boolean allowAnyTreeArtifacts)
      throws EvalException {
    for (Artifact file : files) {
      if (allowAnyTreeArtifacts && file.isTreeArtifact()) {
        continue;
      }
      if (!validFileTypeSet.matches(file.getFilename())) {
        throw Starlark.errorf(
            "'%s' has wrong extension. The list of possible extensions for '%s' is: %s",
            file.getExecPathString(),
            paramName,
            Joiner.on(",").join(fileTypeForErrorMessage.getExtensions()));
      }
    }
  }

  @StarlarkMethod(
      name = "register_linkstamp_compile_action",
      documented = false,
      useStarlarkThread = true,
      parameters = {
        @Param(
            name = "actions",
            positional = false,
            named = true,
            doc = "<code>actions</code> object."),
        @Param(
            name = "cc_toolchain",
            doc = "<code>CcToolchainInfo</code> provider to be used.",
            positional = false,
            named = true),
        @Param(
            name = "feature_configuration",
            doc = "<code>feature_configuration</code> to be queried.",
            positional = false,
            named = true),
        @Param(name = "source_file", documented = false, positional = false, named = true),
        @Param(name = "output_file", documented = false, positional = false, named = true),
        @Param(name = "compilation_inputs", documented = false, positional = false, named = true),
        @Param(
            name = "inputs_for_validation",
            documented = false,
            positional = false,
            named = true),
        @Param(name = "label_replacement", documented = false, positional = false, named = true),
        @Param(name = "output_replacement", documented = false, positional = false, named = true),
        @Param(
            name = "needs_pic",
            documented = false,
            positional = false,
            named = true,
            defaultValue = "False"),
        @Param(
            name = "stamping",
            documented = false,
            positional = false,
            named = true,
            defaultValue = "None"),
        @Param(
            name = "additional_linkstamp_defines",
            positional = false,
            named = true,
            documented = false,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            defaultValue = "unbound"),
      })
  public void registerLinkstampCompileAction(
      StarlarkActionFactory starlarkActionFactoryApi,
      Info ccToolchainInfo,
      FeatureConfigurationForStarlark featureConfigurationForStarlark,
      Artifact sourceFile,
      Artifact outputFile,
      Depset compilationInputs,
      Depset inputsForValidation,
      String labelReplacement,
      String outputReplacement,
      boolean needsPic,
      Object stampingObject,
      Object additionalLinkstampDefines,
      StarlarkThread thread)
      throws EvalException, InterruptedException, TypeException, RuleErrorException {
    isCalledFromStarlarkCcCommon(thread);
    RuleContext ruleContext = starlarkActionFactoryApi.getRuleContext();
    boolean stamping =
        stampingObject instanceof Boolean
            ? (Boolean) stampingObject
            : AnalysisUtils.isStampingEnabled(ruleContext, ruleContext.getConfiguration());
    CcToolchainProvider ccToolchain =
        CcToolchainProvider.PROVIDER.wrapOrThrowEvalException(ccToolchainInfo);
    CppConfiguration cppConfiguration = ccToolchain.getCppConfiguration();
    if (AnalysisUtils.isStampingEnabled(ruleContext, ruleContext.getConfiguration())) {
      // Makes the target depend on BUILD_INFO_KEY, which helps to discover stamped targets
      // See b/326620485 for more details.
      var unused =
          starlarkActionFactoryApi
              .getRuleContext()
              .getAnalysisEnvironment()
              .getVolatileWorkspaceStatusArtifact();
    }
    starlarkActionFactoryApi.registerAction(
        CppLinkstampCompileHelper.createLinkstampCompileAction(
            CppLinkActionBuilder.newActionConstruction(ruleContext),
            sourceFile,
            outputFile,
            compilationInputs.getSet(Artifact.class),
            /* nonCodeInputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            inputsForValidation.getSet(Artifact.class),
            stamping
                ? ccToolchain
                    .getCcBuildInfoTranslator()
                    .getOutputGroup("non_redacted_build_info_files")
                    .toList()
                : ccToolchain
                    .getCcBuildInfoTranslator()
                    .getOutputGroup("redacted_build_info_files")
                    .toList(),
            asStringImmutableList(additionalLinkstampDefines),
            ccToolchain,
            ruleContext.getConfiguration().isCodeCoverageEnabled(),
            CppHelper.getFdoBuildStamp(
                cppConfiguration,
                ccToolchain.getFdoContext(),
                featureConfigurationForStarlark.getFeatureConfiguration()),
            featureConfigurationForStarlark.getFeatureConfiguration(),
            needsPic,
            labelReplacement,
            outputReplacement,
            getSemantics()));
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
      name = "get_cpp_semantics",
      doc = "Gets a CppSemantics object from a language string, for creating Cpp actions.",
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
