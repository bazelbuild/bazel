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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.SkylarkInfo;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper.LinkingInfo;
import com.google.devtools.build.lib.rules.cpp.CcModule.CcSkylarkInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ArtifactNamePattern;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvEntry;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.EnvSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Flag;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagGroup;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FlagSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.VariableWithValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.WithFeatureSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.Expandable;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.StringValueParser;
import com.google.devtools.build.lib.rules.cpp.CppActionConfigs.CppPlatform;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcModuleApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcSkylarkInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import javax.annotation.Nullable;

/** A module that contains Skylark utilities for C++ support. */
public class CcModule
    implements CcModuleApi<
        CcToolchainProvider,
        FeatureConfiguration,
        CcToolchainVariables,
        LibraryToLink,
        CcLinkParams,
        CcSkylarkInfo> {

  private enum RegisterActions {
    ALWAYS,
    NEVER,
    CONDITIONALLY;

    private final String skylarkName;

    RegisterActions() {
      this.skylarkName = toString().toLowerCase();
    }

    public String getSkylarkName() {
      return skylarkName;
    }

    public static RegisterActions fromString(
        String skylarkName, Location location, String fieldForError) throws EvalException {
      for (RegisterActions registerActions : values()) {
        if (registerActions.getSkylarkName().equals(skylarkName)) {
          return registerActions;
        }
      }
      throw new EvalException(
          location,
          String.format(
              "Possibles values for %s: %s",
              fieldForError,
              Joiner.on(", ")
                  .join(
                      Arrays.stream(values())
                          .map(RegisterActions::getSkylarkName)
                          .collect(ImmutableList.toImmutableList()))));
    }
  }

  /**
   * In the rule definition of C++ rules, the only requirement that exists now for a target to be
   * allowed in the list of dependencies is for the target to provide CcInfo. However, there are
   * several native rules that provide CcInfo which were never intended to be dependencies of C++
   * rules, e.g.: Java rules. These rules must wrap the C++ provider in a different one, for now we
   * check in analysis that these rules are not in deps. We mark them by having them provide {@link
   * NonCcDepInfo}
   *
   * <p>TODO(b/77669139): Wrap C++ providers for rules that shouldn't be in deps.
   */
  @Immutable
  @AutoCodec
  public static final class NonCcDepInfo extends NativeInfo {
    public static final ObjectCodec<CcSkylarkInfo> CODEC = new CcModule_CcSkylarkInfo_AutoCodec();

    public static final NativeProvider<NonCcDepInfo> PROVIDER =
        new NativeProvider<NonCcDepInfo>(NonCcDepInfo.class, "NonCcDepInfo") {};

    @AutoCodec.Instantiator
    @VisibleForSerialization
    public NonCcDepInfo() {
      super(PROVIDER);
    }
  }

  /** TODO(b/119754358): Remove this provider after all Skylark rules have stopped using it. */
  @Immutable
  @AutoCodec
  public static final class CcSkylarkInfo extends NativeInfo implements CcSkylarkInfoApi {
    public static final ObjectCodec<CcSkylarkInfo> CODEC = new CcModule_CcSkylarkInfo_AutoCodec();

    public static final NativeProvider<CcSkylarkInfo> PROVIDER =
        new NativeProvider<CcSkylarkInfo>(CcSkylarkInfo.class, "CcSkylarkInfo") {};

    @AutoCodec.Instantiator
    @VisibleForSerialization
    CcSkylarkInfo() {
      super(PROVIDER);
    }
  }

  @Override
  public Provider getCcToolchainProvider() {
    return ToolchainInfo.PROVIDER;
  }

  @Override
  public FeatureConfiguration configureFeatures(
      CcToolchainProvider toolchain,
      SkylarkList<String> requestedFeatures,
      SkylarkList<String> unsupportedFeatures)
      throws EvalException {
    return CcCommon.configureFeaturesOrThrowEvalException(
        ImmutableSet.copyOf(requestedFeatures),
        ImmutableSet.copyOf(unsupportedFeatures),
        toolchain);
  }

  @Override
  public String getToolForAction(FeatureConfiguration featureConfiguration, String actionName) {
    return featureConfiguration.getToolPathForAction(actionName);
  }

  @Override
  public boolean isEnabled(FeatureConfiguration featureConfiguration, String featureName) {
    return featureConfiguration.isEnabled(featureName);
  }

  @Override
  public SkylarkList<String> getCommandLine(
      FeatureConfiguration featureConfiguration,
      String actionName,
      CcToolchainVariables variables) {
    return SkylarkList.createImmutable(featureConfiguration.getCommandLine(actionName, variables));
  }

  @Override
  public SkylarkDict<String, String> getEnvironmentVariable(
      FeatureConfiguration featureConfiguration,
      String actionName,
      CcToolchainVariables variables) {
    return SkylarkDict.copyOf(
        null, featureConfiguration.getEnvironmentVariables(actionName, variables));
  }

  @Override
  public CcToolchainVariables getCompileBuildVariables(
      CcToolchainProvider ccToolchainProvider,
      FeatureConfiguration featureConfiguration,
      Object sourceFile,
      Object outputFile,
      Object userCompileFlags,
      Object includeDirs,
      Object quoteIncludeDirs,
      Object systemIncludeDirs,
      Object defines,
      boolean usePic,
      boolean addLegacyCxxOptions)
      throws EvalException {
    return CompileBuildVariables.setupVariablesOrThrowEvalException(
        featureConfiguration,
        ccToolchainProvider,
        convertFromNoneable(sourceFile, /* defaultValue= */ null),
        convertFromNoneable(outputFile, /* defaultValue= */ null),
        /* gcnoFile= */ null,
        /* dwoFile= */ null,
        /* ltoIndexingFile= */ null,
        /* includes= */ ImmutableList.of(),
        userFlagsToIterable(ccToolchainProvider.getCppConfiguration(), userCompileFlags),
        /* cppModuleMap= */ null,
        usePic,
        /* fakeOutputFile= */ null,
        /* fdoStamp= */ null,
        /* dotdFileExecPath= */ null,
        /* variablesExtensions= */ ImmutableList.of(),
        /* additionalBuildVariables= */ ImmutableMap.of(),
        /* directModuleMaps= */ ImmutableList.of(),
        asStringNestedSet(includeDirs),
        asStringNestedSet(quoteIncludeDirs),
        asStringNestedSet(systemIncludeDirs),
        asStringNestedSet(defines),
        addLegacyCxxOptions);
  }

  @Override
  public CcToolchainVariables getLinkBuildVariables(
      CcToolchainProvider ccToolchainProvider,
      FeatureConfiguration featureConfiguration,
      Object librarySearchDirectories,
      Object runtimeLibrarySearchDirectories,
      Object userLinkFlags,
      Object outputFile,
      Object paramFile,
      Object defFile,
      boolean isUsingLinkerNotArchiver,
      boolean isCreatingSharedLibrary,
      boolean mustKeepDebug,
      boolean useTestOnlyFlags,
      boolean isStaticLinkingMode)
      throws EvalException {
    return LinkBuildVariables.setupVariables(
        isUsingLinkerNotArchiver,
        /* binDirectoryPath= */ null,
        convertFromNoneable(outputFile, /* defaultValue= */ null),
        isCreatingSharedLibrary,
        convertFromNoneable(paramFile, /* defaultValue= */ null),
        /* thinltoParamFile= */ null,
        /* thinltoMergedObjectFile= */ null,
        mustKeepDebug,
        /* symbolCounts= */ null,
        ccToolchainProvider,
        featureConfiguration,
        useTestOnlyFlags,
        /* isLtoIndexing= */ false,
        userFlagsToIterable(ccToolchainProvider.getCppConfiguration(), userLinkFlags),
        /* interfaceLibraryBuilder= */ null,
        /* interfaceLibraryOutput= */ null,
        /* ltoOutputRootPrefix= */ null,
        convertFromNoneable(defFile, /* defaultValue= */ null),
        /* fdoProvider= */ null,
        asStringNestedSet(runtimeLibrarySearchDirectories),
        /* librariesToLink= */ null,
        asStringNestedSet(librarySearchDirectories),
        /* isLegacyFullyStaticLinkingMode= */ false,
        isStaticLinkingMode,
        /* addIfsoRelatedVariables= */ false);
  }

  @Override
  public CcToolchainVariables getVariables() {
    return CcToolchainVariables.EMPTY;
  }

  /**
   * Converts an object that can be the NoneType to the actual object if it is not or returns the
   * default value if none.
   */
  @SuppressWarnings("unchecked")
  protected static <T> T convertFromNoneable(Object obj, @Nullable T defaultValue) {
    if (EvalUtils.isNullOrNone(obj)) {
      return defaultValue;
    }
    return (T) obj;
  }

  /** Converts an object that can be ether SkylarkNestedSet or None into NestedSet. */
  protected NestedSet<String> asStringNestedSet(Object o) {
    SkylarkNestedSet skylarkNestedSet = convertFromNoneable(o, /* defaultValue= */ null);
    if (skylarkNestedSet != null) {
      return skylarkNestedSet.getSet(String.class);
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /** Converts an object that can be either SkylarkList, or None into ImmutableList. */
  protected ImmutableList<String> asStringImmutableList(Object o) {
    SkylarkList skylarkList = convertFromNoneable(o, /* defaultValue= */ null);
    if (skylarkList != null) {
      return skylarkList.getImmutableList();
    } else {
      return ImmutableList.of();
    }
  }

  /**
   * Converts an object that represents user flags and can be either SkylarkNestedSet , SkylarkList,
   * or None into Iterable.
   */
  protected Iterable<String> userFlagsToIterable(CppConfiguration cppConfiguration, Object o)
      throws EvalException {
    if (o instanceof SkylarkNestedSet) {
      if (cppConfiguration.disableDepsetInUserFlags()) {
        throw new EvalException(
            Location.BUILTIN,
            "Passing depset into user flags is deprecated (see "
                + "--incompatible_disable_depset_in_cc_user_flags), use list instead.");
      }
      return asStringNestedSet(o);
    } else if (o instanceof SkylarkList) {
      return asStringImmutableList(o);
    } else if (o instanceof NoneType) {
      return ImmutableList.of();
    } else {
      throw new EvalException(Location.BUILTIN, "Only depset and list is allowed.");
    }
  }

  @Override
  public LibraryToLink createLibraryLinkerInput(
      SkylarkRuleContext skylarkRuleContext, Artifact library, String skylarkArtifactCategory)
      throws EvalException, InterruptedException {
    CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    ArtifactCategory artifactCategory =
        ArtifactCategory.fromString(
            skylarkArtifactCategory,
            skylarkRuleContext.getRuleContext().getRule().getLocation(),
            "artifact_category");
    return LinkerInputs.opaqueLibraryToLink(
        library, artifactCategory, CcLinkingOutputs.libraryIdentifierOf(library));
  }

  @Override
  public LibraryToLink createSymlinkLibraryLinkerInput(
      SkylarkRuleContext skylarkRuleContext, CcToolchainProvider ccToolchain, Artifact library) {
    Artifact dynamicLibrarySymlink =
        SolibSymlinkAction.getDynamicLibrarySymlink(
            /* actionRegistry= */ skylarkRuleContext.getRuleContext(),
            /* actionConstructionContext= */ skylarkRuleContext.getRuleContext(),
            ccToolchain.getSolibDirectory(),
            library,
            /* preserveName= */ true,
            /* prefixConsumer= */ true,
            skylarkRuleContext.getRuleContext().getConfiguration());
    return LinkerInputs.solibLibraryToLink(
        dynamicLibrarySymlink, library, CcLinkingOutputs.libraryIdentifierOf(library));
  }

  @Override
  public CcLinkParams createCcLinkParams(
      SkylarkRuleContext skylarkRuleContext,
      Object skylarkLibrariesToLink,
      Object skylarkDynamicLibrariesForRuntime,
      Object skylarkUserLinkFlags)
      throws EvalException, InterruptedException {
    CcCommon.checkRuleWhitelisted(skylarkRuleContext);

    SkylarkNestedSet librariesToLink = convertFromNoneable(skylarkLibrariesToLink, null);
    SkylarkNestedSet dynamicLibrariesForRuntime =
        convertFromNoneable(skylarkDynamicLibrariesForRuntime, null);
    SkylarkNestedSet userLinkFlags = convertFromNoneable(skylarkUserLinkFlags, null);

    CcLinkParams.Builder builder = CcLinkParams.builder();
    if (librariesToLink != null) {
      builder.addLibraries(librariesToLink.toCollection(LibraryToLink.class));
    }
    if (dynamicLibrariesForRuntime != null) {
      builder.addDynamicLibrariesForRuntime(
          dynamicLibrariesForRuntime.toCollection(Artifact.class));
    }
    if (userLinkFlags != null) {
      builder.addLinkOpts(userLinkFlags.toCollection(String.class));
    }
    return builder.build();
  }

  @Override
  public CcSkylarkInfo createCcSkylarkInfo(Object skylarkRuleContextObject)
      throws EvalException, InterruptedException {
    SkylarkRuleContext skylarkRuleContext =
        convertFromNoneable(skylarkRuleContextObject, /* defaultValue= */ null);
    if (skylarkRuleContext != null) {
      CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    }
    return new CcSkylarkInfo();
  }

  @SkylarkCallable(
      name = "merge_cc_infos",
      documented = false,
      parameters = {
        @Param(
            name = "cc_infos",
            doc = "cc_infos to be merged.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class)
      })
  public CcInfo mergeCcInfos(SkylarkList<CcInfo> ccInfos) {
    return CcInfo.merge(ccInfos);
  }

  @SkylarkCallable(
      name = "create_compilation_context",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "headers",
            doc = "the set of headers needed to compile this target",
            positional = false,
            named = true,
            defaultValue = "depset([])",
            type = SkylarkNestedSet.class),
        @Param(
            name = "system_includes",
            doc =
                "set of  search paths for headers file referenced by angle brackets, i.e. "
                    + "<header>.They can be either relative to the exec root or absolute",
            positional = false,
            named = true,
            defaultValue = "depset([])",
            type = SkylarkNestedSet.class),
        @Param(
            name = "defines",
            doc = "the set of defines needed to compile this target. Each define is a string",
            positional = false,
            named = true,
            defaultValue = "depset([])",
            type = SkylarkNestedSet.class)
      })
  public CcCompilationContext createCcCompilationContext(
      SkylarkRuleContext skylarkRuleContext,
      SkylarkNestedSet headers,
      SkylarkNestedSet systemIncludes,
      SkylarkNestedSet defines)
      throws EvalException, InterruptedException {
    CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    CcCompilationContext.Builder ccCompilationContext =
        new CcCompilationContext.Builder(/* ruleContext= */ null);
    ccCompilationContext.addDeclaredIncludeSrcs(headers.getSet(Artifact.class));
    ccCompilationContext.addSystemIncludeDirs(
        systemIncludes.getSet(String.class).toList().stream()
            .map(x -> PathFragment.create(x))
            .collect(ImmutableList.toImmutableList()));
    ccCompilationContext.addDefines(defines.getSet(String.class));
    return ccCompilationContext.build();
  }

  @SkylarkCallable(
      name = "create_linking_context",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "static_mode_params_for_dynamic_library",
            positional = false,
            named = true,
            type = CcLinkParams.class,
            doc = "Parameters for linking a dynamic library statically."),
        @Param(
            name = "static_mode_params_for_executable",
            doc = "Parameters for linking an executable statically",
            positional = false,
            named = true,
            type = CcLinkParams.class),
        @Param(
            name = "dynamic_mode_params_for_dynamic_library",
            doc = "Parameters for linking a dynamic library dynamically",
            positional = false,
            named = true,
            type = CcLinkParams.class),
        @Param(
            name = "dynamic_mode_params_for_executable",
            doc = "Parameters for linking an executable dynamically",
            positional = false,
            named = true,
            type = CcLinkParams.class)
      })
  public CcLinkingInfo createCcLinkingInfo(
      SkylarkRuleContext skylarkRuleContext,
      CcLinkParams staticModeParamsForDynamicLibrary,
      CcLinkParams staticModeParamsForExecutable,
      CcLinkParams dynamicModeParamsForDynamicLibrary,
      CcLinkParams dynamicModeParamsForExecutable)
      throws EvalException, InterruptedException {
    CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    CcLinkingInfo.Builder ccLinkingInfoBuilder = CcLinkingInfo.Builder.create();
    ccLinkingInfoBuilder
        .setStaticModeParamsForDynamicLibrary(staticModeParamsForDynamicLibrary)
        .setStaticModeParamsForExecutable(staticModeParamsForExecutable)
        .setDynamicModeParamsForDynamicLibrary(dynamicModeParamsForDynamicLibrary)
        .setDynamicModeParamsForExecutable(dynamicModeParamsForExecutable);
    return ccLinkingInfoBuilder.build();
  }

  protected static CompilationInfo compile(
      CppSemantics cppSemantics,
      SkylarkRuleContext skylarkRuleContext,
      Object skylarkFeatureConfiguration,
      Object skylarkCcToolchainProvider,
      SkylarkList<Artifact> sources,
      SkylarkList<Artifact> headers,
      Object skylarkIncludes,
      Object skylarkCopts,
      String generateNoPicOutputs,
      String generatePicOutputs,
      Object skylarkAdditionalCompilationInputs,
      Object skylarkAdditionalIncludeScanningRoots,
      SkylarkList<CcCompilationContext> ccCompilationContexts,
      Object purpose)
      throws EvalException, InterruptedException {
    CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    RuleContext ruleContext = skylarkRuleContext.getRuleContext();
    CcToolchainProvider ccToolchainProvider = convertFromNoneable(skylarkCcToolchainProvider, null);
    FeatureConfiguration featureConfiguration =
        convertFromNoneable(skylarkFeatureConfiguration, null);
    Pair<List<Artifact>, List<Artifact>> separatedHeadersAndSources =
        separateSourcesFromHeaders(sources);
    FdoProvider fdoProvider = ccToolchainProvider.getFdoProvider();
    // TODO(plf): Need to flatten the nested set to convert the Strings to PathFragment. This could
    // be avoided if path fragments are ever added to Skylark or in the C++ code we take Strings
    // instead of PathFragments.
    List<String> includeDirs = convertSkylarkListOrNestedSetToList(skylarkIncludes, String.class);
    CcCompilationHelper helper =
        new CcCompilationHelper(
                ruleContext,
                cppSemantics,
                featureConfiguration,
                CcCompilationHelper.SourceCategory.CC,
                ccToolchainProvider,
                fdoProvider)
            .addPublicHeaders(headers)
            .addIncludeDirs(
                includeDirs.stream()
                    .map(PathFragment::create)
                    .collect(ImmutableList.toImmutableList()))
            .addPrivateHeaders(separatedHeadersAndSources.first)
            .addSources(separatedHeadersAndSources.second)
            .addCcCompilationContexts(ccCompilationContexts)
            .setPurpose(convertFromNoneable(purpose, null));

    SkylarkNestedSet additionalCompilationInputs =
        convertFromNoneable(skylarkAdditionalCompilationInputs, null);
    if (additionalCompilationInputs != null) {
      helper.addAdditionalCompilationInputs(
          additionalCompilationInputs.toCollection(Artifact.class));
    }

    SkylarkNestedSet additionalIncludeScanningRoots =
        convertFromNoneable(skylarkAdditionalIncludeScanningRoots, null);
    if (additionalIncludeScanningRoots != null) {
      helper.addAditionalIncludeScanningRoots(
          additionalIncludeScanningRoots.toCollection(Artifact.class));
    }

    SkylarkNestedSet copts = convertFromNoneable(skylarkCopts, null);
    if (copts != null) {
      helper.setCopts(copts.getSet(String.class));
    }

    Location location = ruleContext.getRule().getLocation();
    RegisterActions generateNoPicOption =
        RegisterActions.fromString(generateNoPicOutputs, location, "generate_no_pic_outputs");
    if (!generateNoPicOption.equals(RegisterActions.CONDITIONALLY)) {
      helper.setGenerateNoPicAction(generateNoPicOption == RegisterActions.ALWAYS);
    }
    RegisterActions generatePicOption =
        RegisterActions.fromString(generatePicOutputs, location, "generate_pic_outputs");
    if (!generatePicOption.equals(RegisterActions.CONDITIONALLY)) {
      helper.setGeneratePicAction(generatePicOption == RegisterActions.ALWAYS);
    }
    try {
      return helper.compile();
    } catch (RuleErrorException e) {
      throw new EvalException(ruleContext.getRule().getLocation(), e);
    }
  }

  protected static LinkingInfo link(
      CppSemantics cppSemantics,
      SkylarkRuleContext skylarkRuleContext,
      Object skylarkFeatureConfiguration,
      Object skylarkCcToolchainProvider,
      CcCompilationOutputs ccCompilationOutputs,
      Object skylarkLinkopts,
      boolean shouldCreateStaticLibraries,
      Object dynamicLibrary,
      SkylarkList<CcLinkingInfo> skylarkCcLinkingInfos,
      boolean neverLink)
      throws InterruptedException, EvalException, InterruptedException {
    CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    RuleContext ruleContext = skylarkRuleContext.getRuleContext();
    CcToolchainProvider ccToolchainProvider = convertFromNoneable(skylarkCcToolchainProvider, null);
    FeatureConfiguration featureConfiguration =
        convertFromNoneable(skylarkFeatureConfiguration, null);
    FdoProvider fdoProvider = ccToolchainProvider.getFdoProvider();
    NestedSet<String> linkopts =
        convertSkylarkListOrNestedSetToNestedSet(skylarkLinkopts, String.class);
    CcLinkingHelper helper =
        new CcLinkingHelper(
                ruleContext,
                cppSemantics,
                featureConfiguration,
                ccToolchainProvider,
                fdoProvider,
                ruleContext.getConfiguration())
            .addLinkopts(linkopts)
            .setShouldCreateStaticLibraries(shouldCreateStaticLibraries)
            .setLinkerOutputArtifact(convertFromNoneable(dynamicLibrary, null))
            .addCcLinkingInfos(skylarkCcLinkingInfos)
            .setNeverLink(neverLink);
    try {
      CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
      if (!ccCompilationOutputs.isEmpty()) {
        ccLinkingOutputs = helper.link(ccCompilationOutputs);
      }
      CcLinkingInfo ccLinkingInfo =
          helper.buildCcLinkingInfo(ccLinkingOutputs, CcCompilationContext.EMPTY);
      return new LinkingInfo(ccLinkingInfo, ccLinkingOutputs);
    } catch (RuleErrorException e) {
      throw new EvalException(ruleContext.getRule().getLocation(), e);
    }
  }

  /**
   * TODO(plf): This method exists only temporarily. Once the existing C++ rules have been migrated,
   * they should pass sources and headers separately.
   */
  private static Pair<List<Artifact>, List<Artifact>> separateSourcesFromHeaders(
      Iterable<Artifact> artifacts) {
    List<Artifact> headers = new ArrayList<>();
    List<Artifact> sources = new ArrayList<>();
    for (Artifact artifact : artifacts) {
      if (CppFileTypes.CPP_HEADER.matches(artifact.getExecPath())) {
        headers.add(artifact);
      } else {
        sources.add(artifact);
      }
    }
    return Pair.of(headers, sources);
  }

  /** Converts an object that can be the either SkylarkNestedSet or None into NestedSet. */
  @SuppressWarnings("unchecked")
  protected Object skylarkListToSkylarkNestedSet(Object o) throws EvalException {
    if (o instanceof SkylarkList) {
      SkylarkList<String> list = (SkylarkList<String>) o;
      SkylarkNestedSet.Builder builder =
          SkylarkNestedSet.builder(Order.STABLE_ORDER, Location.BUILTIN);
      for (Object entry : list) {
        builder.addDirect(entry);
      }
      return builder.build();
    }
    return o;
  }

  @SuppressWarnings("unchecked")
  private static <T> List<T> convertSkylarkListOrNestedSetToList(Object o, Class<T> type) {
    return o instanceof SkylarkNestedSet
        ? ((SkylarkNestedSet) o).getSet(type).toList()
        : ((SkylarkList) o).getImmutableList();
  }

  @SuppressWarnings("unchecked")
  private static <T> NestedSet<T> convertSkylarkListOrNestedSetToNestedSet(
      Object o, Class<T> type) {
    return o instanceof SkylarkNestedSet
        ? ((SkylarkNestedSet) o).getSet(type)
        : NestedSetBuilder.wrap(Order.COMPILE_ORDER, (SkylarkList<T>) o);
  }

  @SkylarkCallable(
      name = "create_cc_toolchain_config_info",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "features",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "action_configs",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "artifact_name_patterns",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "cxx_builtin_include_directories",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "toolchain_identifier",
            positional = false,
            type = String.class,
            named = true),
        @Param(name = "host_system_name", positional = false, type = String.class, named = true),
        @Param(name = "target_system_name", positional = false, type = String.class, named = true),
        @Param(name = "target_cpu", positional = false, type = String.class, named = true),
        @Param(name = "target_libc", positional = false, type = String.class, named = true),
        @Param(name = "compiler", positional = false, type = String.class, named = true),
        @Param(name = "abi_version", positional = false, type = String.class, named = true),
        @Param(name = "abi_libc_version", positional = false, type = String.class, named = true),
        @Param(
            name = "supports_gold_linker",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true),
        @Param(
            name = "supports_start_end_lib",
            positional = false,
            type = Boolean.class,
            defaultValue = "False",
            named = true),
        @Param(
            name = "supports_interface_shared_objects",
            positional = false,
            type = Boolean.class,
            defaultValue = "False",
            named = true),
        @Param(
            name = "supports_embedded_runtimes",
            positional = false,
            type = Boolean.class,
            defaultValue = "False",
            named = true),
        @Param(
            name = "static_runtime_filegroup",
            positional = false,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            named = true),
        @Param(
            name = "dynamic_runtime_filegroup",
            positional = false,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            named = true),
        @Param(
            name = "supports_fission",
            positional = false,
            type = Boolean.class,
            defaultValue = "False",
            named = true),
        @Param(
            name = "supports_dsym",
            positional = false,
            type = Boolean.class,
            defaultValue = "False",
            named = true),
        @Param(
            name = "needs_pic",
            positional = false,
            type = Boolean.class,
            defaultValue = "False",
            named = true),
        @Param(
            name = "tool_paths",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "compiler_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "cxx_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "unfiltered_cxx_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "linker_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "dynamic_library_linker_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "test_only_linker_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "objcopy_embed_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "ld_embed_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "compilation_mode_compiler_flags",
            positional = false,
            named = true,
            defaultValue = "{}",
            type = SkylarkDict.class),
        @Param(
            name = "compilation_mode_cxx_flags",
            positional = false,
            named = true,
            defaultValue = "{}",
            type = SkylarkDict.class),
        @Param(
            name = "compilation_mode_linker_flags",
            positional = false,
            named = true,
            defaultValue = "{}",
            type = SkylarkDict.class),
        @Param(
            name = "mostly_static_linking_mode_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "dynamic_linking_mode_flags",
            positional = false,
            named = true,
            defaultValue = "None",
            noneable = true,
            allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = NoneType.class)
            }),
        @Param(
            name = "fully_static_linking_mode_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "mostly_static_libraries_linking_mode_flags",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "make_variables",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class),
        @Param(
            name = "builtin_sysroot",
            positional = false,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            named = true),
        @Param(
            name = "default_libc_top",
            positional = false,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            named = true),
        @Param(
            name = "cc_target_os",
            positional = false,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
            named = true),
      })
  public CcToolchainConfigInfo ccToolchainConfigInfoFromSkylark(
      SkylarkRuleContext skylarkRuleContext,
      SkylarkList<Object> features,
      SkylarkList<Object> actionConfigs,
      SkylarkList<Object> artifactNamePatterns,
      SkylarkList<String> cxxBuiltInIncludeDirectories,
      String toolchainIdentifier,
      String hostSystemName,
      String targetSystemName,
      String targetCpu,
      String targetLibc,
      String compiler,
      String abiVersion,
      String abiLibcVersion,
      Boolean supportsGoldLinker,
      Boolean supportsStartEndLib,
      Boolean supportsInterfaceSharedObjects,
      Boolean supportsEmbeddedRuntimes,
      Object staticRuntimesFilegroup,
      Object dynamicRuntimesFilegroup,
      Boolean supportsFission,
      Boolean supportsDsym,
      Boolean needsPic,
      SkylarkList<Object> toolPaths,
      SkylarkList<String> compilerFlags,
      SkylarkList<String> cxxFlags,
      SkylarkList<String> unfilteredCxxFlags,
      SkylarkList<String> linkerFlags,
      SkylarkList<String> dynamicLibraryLinkerFlags,
      SkylarkList<String> testOnlyLinkerFlags,
      SkylarkList<String> objcopyEmbedFlags,
      SkylarkList<String> ldEmbedFlags,
      Object compilationModeCompilerFlagsUnchecked,
      Object compilationModeCxxFlagsUnchecked,
      Object compilationModeLinkerFlagsUnchecked,
      SkylarkList<String> mostlyStaticLinkingModeFlags,
      Object dynamicLinkingModeFlags,
      SkylarkList<String> fullyStaticLinkingModeFlags,
      SkylarkList<String> mostlyStaticLibrariesLinkingModeFlags,
      SkylarkList<Object> makeVariables,
      Object builtinSysroot,
      Object defaultLibcTop,
      Object ccTargetOs)
      throws InvalidConfigurationException, EvalException {

    CppConfiguration config =
        skylarkRuleContext.getConfiguration().getFragment(CppConfiguration.class);
    if (!config.enableCcToolchainConfigInfoFromSkylark()) {
      throw new InvalidConfigurationException("Creating a CcToolchainConfigInfo is not enabled.");
    }
    if (!config.disableMakeVariables()) {
      throw new InvalidConfigurationException(
          "--incompatible_disable_cc_configuration_make_variables must be set to true in "
              + "order to configure the C++ toolchain from Starlark.");
    }

    ImmutableList.Builder<Feature> featureBuilder = ImmutableList.builder();
    for (Object feature : features) {
      featureBuilder.add(featureFromSkylark((SkylarkInfo) feature));
    }
    ImmutableList<Feature> featureList = featureBuilder.build();

    ImmutableSet<String> featureNames =
        featureList.stream()
            .map(feature -> feature.getName())
            .collect(ImmutableSet.toImmutableSet());

    ImmutableList.Builder<ActionConfig> actionConfigBuilder = ImmutableList.builder();
    for (Object actionConfig : actionConfigs) {
      actionConfigBuilder.add(actionConfigFromSkylark((SkylarkInfo) actionConfig));
    }
    ImmutableList<ActionConfig> actionConfigList = actionConfigBuilder.build();

    ImmutableList.Builder<ArtifactNamePattern> artifactNamePatternBuilder = ImmutableList.builder();
    for (Object artifactNamePattern : artifactNamePatterns) {
      artifactNamePatternBuilder.add(
          artifactNamePatternFromSkylark((SkylarkInfo) artifactNamePattern));
    }
    getLegacyArtifactNamePatterns(artifactNamePatternBuilder);

    // Pairs (toolName, toolPath)
    ImmutableList.Builder<Pair<String, String>> toolPathPairs = ImmutableList.builder();
    for (Object toolPath : toolPaths) {
      toolPathPairs.add(toolPathFromSkylark((SkylarkInfo) toolPath));
    }
    ImmutableList<Pair<String, String>> toolPathList = toolPathPairs.build();

    if (!featureNames.contains(CppRuleClasses.NO_LEGACY_FEATURES)) {
      String gccToolPath = "DUMMY_GCC_TOOL";
      String linkerToolPath = "DUMMY_LINKER_TOOL";
      String arToolPath = "DUMMY_AR_TOOL";
      String stripToolPath = "DUMMY_STRIP_TOOL";
      for (Pair<String, String> tool : toolPathList) {
        if (tool.first.equals(CppConfiguration.Tool.GCC.getNamePart())) {
          gccToolPath = tool.second;
          linkerToolPath =
              skylarkRuleContext
                  .getRuleContext()
                  .getLabel()
                  .getPackageIdentifier()
                  .getPathUnderExecRoot()
                  .getRelative(PathFragment.create(tool.second))
                  .getPathString();
        }
        if (tool.first.equals(CppConfiguration.Tool.AR.getNamePart())) {
          arToolPath = tool.second;
        }
        if (tool.first.equals(CppConfiguration.Tool.STRIP.getNamePart())) {
          stripToolPath = tool.second;
        }
      }

      ImmutableList.Builder<Feature> legacyFeaturesBuilder = ImmutableList.builder();
      // TODO(b/30109612): Remove fragile legacyCompileFlags shuffle once there are no legacy
      // crosstools.
      // Existing projects depend on flags from legacy toolchain fields appearing first on the
      // compile command line. 'legacy_compile_flags' feature contains all these flags, and so it
      // needs to appear before other features from {@link CppActionConfigs}.
      if (featureNames.contains(CppRuleClasses.LEGACY_COMPILE_FLAGS)) {
        legacyFeaturesBuilder.add(
            featureList.stream()
                .filter(feature -> feature.getName().equals(CppRuleClasses.LEGACY_COMPILE_FLAGS))
                .findFirst()
                .get());
      }

      CppPlatform platform = targetLibc.equals("macos") ? CppPlatform.MAC : CppPlatform.LINUX;
      for (CToolchain.Feature feature :
          CppActionConfigs.getLegacyFeatures(
              platform,
              featureNames,
              linkerToolPath,
              // This should be toolchain-based, rather than feature based, because
              // it controls whether or not to declare the feature at all.
              supportsEmbeddedRuntimes,
              supportsInterfaceSharedObjects)) {
        legacyFeaturesBuilder.add(new Feature(feature));
      }
      legacyFeaturesBuilder.addAll(
          featureList.stream()
              .filter(feature -> !feature.getName().equals(CppRuleClasses.LEGACY_COMPILE_FLAGS))
              .collect(ImmutableList.toImmutableList()));
      for (CToolchain.Feature feature :
          CppActionConfigs.getFeaturesToAppearLastInFeaturesList(featureNames)) {
        legacyFeaturesBuilder.add(new Feature(feature));
      }

      featureList = legacyFeaturesBuilder.build();

      ImmutableList.Builder<ActionConfig> legacyActionConfigBuilder = ImmutableList.builder();
      for (CToolchain.ActionConfig actionConfig :
          CppActionConfigs.getLegacyActionConfigs(
              platform, gccToolPath, arToolPath, stripToolPath, supportsEmbeddedRuntimes)) {
        legacyActionConfigBuilder.add(new ActionConfig(actionConfig));
      }
      legacyActionConfigBuilder.addAll(actionConfigList);
      actionConfigList = legacyActionConfigBuilder.build();
    }

    ImmutableList.Builder<Pair<String, String>> makeVariablePairs = ImmutableList.builder();
    for (Object makeVariable : makeVariables) {
      makeVariablePairs.add(makeVariableFromSkylark((SkylarkInfo) makeVariable));
    }

    SkylarkList<String> dynamicModeFlags =
        convertFromNoneable(dynamicLinkingModeFlags, /* defaultValue= */ null);
    boolean hasDynamicLinkingModeFlags = dynamicModeFlags != null;

    return new CcToolchainConfigInfo(
        actionConfigList,
        featureList,
        artifactNamePatternBuilder.build(),
        ImmutableList.copyOf(cxxBuiltInIncludeDirectories),
        toolchainIdentifier,
        hostSystemName,
        targetSystemName,
        targetCpu,
        targetLibc,
        compiler,
        abiVersion,
        abiLibcVersion,
        supportsGoldLinker,
        supportsStartEndLib,
        supportsInterfaceSharedObjects,
        supportsEmbeddedRuntimes,
        convertFromNoneable(staticRuntimesFilegroup, /* defaultValue= */ ""),
        convertFromNoneable(dynamicRuntimesFilegroup, /* defaultValue= */ ""),
        supportsFission,
        supportsDsym,
        needsPic,
        toolPathList,
        ImmutableList.copyOf(compilerFlags),
        ImmutableList.copyOf(cxxFlags),
        ImmutableList.copyOf(unfilteredCxxFlags),
        ImmutableList.copyOf(linkerFlags),
        ImmutableList.copyOf(dynamicLibraryLinkerFlags),
        ImmutableList.copyOf(testOnlyLinkerFlags),
        ImmutableList.copyOf(objcopyEmbedFlags),
        ImmutableList.copyOf(ldEmbedFlags),
        getCompilationModeFlagsFromSkylark(
            compilationModeCompilerFlagsUnchecked, "compilation_mode_compiler_flags"),
        getCompilationModeFlagsFromSkylark(
            compilationModeCxxFlagsUnchecked, "compilation_mode_cxx_flags"),
        getCompilationModeFlagsFromSkylark(
            compilationModeLinkerFlagsUnchecked, "compilation_mode_linker_flags"),
        ImmutableList.copyOf(mostlyStaticLinkingModeFlags),
        hasDynamicLinkingModeFlags ? ImmutableList.copyOf(dynamicModeFlags) : ImmutableList.of(),
        ImmutableList.copyOf(fullyStaticLinkingModeFlags),
        ImmutableList.copyOf(mostlyStaticLibrariesLinkingModeFlags),
        makeVariablePairs.build(),
        convertFromNoneable(builtinSysroot, /* defaultValue= */ ""),
        convertFromNoneable(defaultLibcTop, /* defaultValue= */ ""),
        convertFromNoneable(ccTargetOs, /* defaultValue= */ ""),
        hasDynamicLinkingModeFlags);
  }

  /** Checks whether the {@link SkylarkInfo} is of the required type. */
  private static void checkRightProviderType(SkylarkInfo provider, String type)
      throws EvalException {
    String providerType = (String) provider.getValueOrNull("type_name");
    if (providerType == null) {
      providerType = provider.getProvider().getPrintableName();
    }
    if (!provider.hasField("type_name") || !provider.getValue("type_name").equals(type)) {
      throw new EvalException(
          provider.getCreationLoc(),
          String.format("Expected object of type '%s', received '%s'.", type, providerType));
    }
  }

  /** Creates a {@link Feature} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static Feature featureFromSkylark(SkylarkInfo featureStruct)
      throws InvalidConfigurationException, EvalException {
    checkRightProviderType(featureStruct, "feature");
    String name = getFieldFromSkylarkProvider(featureStruct, "name", String.class);
    Boolean enabled = getFieldFromSkylarkProvider(featureStruct, "enabled", Boolean.class);
    if (name == null || (name.isEmpty() && !enabled)) {
      throw new EvalException(
          featureStruct.getCreationLoc(),
          "A feature must either have a nonempty 'name' field or be enabled.");
    }

    if (!name.matches("^[_a-z+\\-]*$")) {
      throw new EvalException(
          featureStruct.getCreationLoc(),
          String.format(
              "A feature's name must consist solely of lowercase ASCII letters, '_', '+', and '-', "
                  + "got '%s'",
              name));
    }

    ImmutableList.Builder<FlagSet> flagSetBuilder = ImmutableList.builder();
    ImmutableList<SkylarkInfo> flagSets =
        getSkylarkProviderListFromSkylarkField(featureStruct, "flag_sets");
    for (SkylarkInfo flagSet : flagSets) {
      flagSetBuilder.add(flagSetFromSkylark(flagSet));
    }

    ImmutableList.Builder<EnvSet> envSetBuilder = ImmutableList.builder();
    ImmutableList<SkylarkInfo> envSets =
        getSkylarkProviderListFromSkylarkField(featureStruct, "env_sets");
    for (SkylarkInfo envSet : envSets) {
      envSetBuilder.add(envSetFromSkylark(envSet));
    }

    ImmutableList.Builder<ImmutableSet<String>> requiresBuilder = ImmutableList.builder();

    ImmutableList<SkylarkInfo> requires =
        getSkylarkProviderListFromSkylarkField(featureStruct, "requires");
    for (SkylarkInfo featureSetStruct : requires) {
      if (!featureSetStruct.hasField("type_name")
          || !featureSetStruct.getValue("type_name").equals("feature_set")) {
        throw new EvalException(
            featureStruct.getCreationLoc(), "expected object of type 'feature_set'.");
      }
      ImmutableSet<String> featureSet =
          getStringSetFromSkylarkProviderField(featureSetStruct, "features");
      requiresBuilder.add(featureSet);
    }

    ImmutableList<String> implies = getStringListFromSkylarkProviderField(featureStruct, "implies");

    ImmutableList<String> provides =
        getStringListFromSkylarkProviderField(featureStruct, "provides");

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
   * SkylarkInfo}.
   */
  @VisibleForTesting
  static Pair<String, String> makeVariableFromSkylark(SkylarkInfo makeVariableStruct)
      throws EvalException {
    checkRightProviderType(makeVariableStruct, "make_variable");
    String name = getFieldFromSkylarkProvider(makeVariableStruct, "name", String.class);
    String value = getFieldFromSkylarkProvider(makeVariableStruct, "value", String.class);
    if (name == null || name.isEmpty()) {
      throw new EvalException(
          makeVariableStruct.getCreationLoc(),
          "'name' parameter of make_variable must be a nonempty string.");
    }
    if (value == null || value.isEmpty()) {
      throw new EvalException(
          makeVariableStruct.getCreationLoc(),
          "'value' parameter of make_variable must be a nonempty string.");
    }
    return Pair.of(name, value);
  }

  /**
   * Creates a Pair(name, path) that represents a {@link
   * com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath} from a {@link
   * SkylarkInfo}.
   */
  @VisibleForTesting
  static Pair<String, String> toolPathFromSkylark(SkylarkInfo toolPathStruct) throws EvalException {
    checkRightProviderType(toolPathStruct, "tool_path");
    String name = getFieldFromSkylarkProvider(toolPathStruct, "name", String.class);
    String path = getFieldFromSkylarkProvider(toolPathStruct, "path", String.class);
    if (name == null || name.isEmpty()) {
      throw new EvalException(
          toolPathStruct.getCreationLoc(),
          "'name' parameter of tool_path must be a nonempty string.");
    }
    if (path == null || path.isEmpty()) {
      throw new EvalException(
          toolPathStruct.getCreationLoc(),
          "'path' parameter of tool_path must be a nonempty string.");
    }
    return Pair.of(name, path);
  }

  /** Creates a {@link VariableWithValue} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static VariableWithValue variableWithValueFromSkylark(SkylarkInfo variableWithValueStruct)
      throws EvalException {
    checkRightProviderType(variableWithValueStruct, "variable_with_value");
    String name = getFieldFromSkylarkProvider(variableWithValueStruct, "name", String.class);
    String value = getFieldFromSkylarkProvider(variableWithValueStruct, "value", String.class);
    if (name == null || name.isEmpty()) {
      throw new EvalException(
          variableWithValueStruct.getCreationLoc(),
          "'name' parameter of variable_with_value must be a nonempty string.");
    }
    if (value == null || value.isEmpty()) {
      throw new EvalException(
          variableWithValueStruct.getCreationLoc(),
          "'value' parameter of variable_with_value must be a nonempty string.");
    }
    return new VariableWithValue(name, value);
  }

  /** Creates an {@link EnvEntry} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static EnvEntry envEntryFromSkylark(SkylarkInfo envEntryStruct)
      throws InvalidConfigurationException, EvalException {
    checkRightProviderType(envEntryStruct, "env_entry");
    String key = getFieldFromSkylarkProvider(envEntryStruct, "key", String.class);
    String value = getFieldFromSkylarkProvider(envEntryStruct, "value", String.class);
    if (key == null || key.isEmpty()) {
      throw new EvalException(
          envEntryStruct.getCreationLoc(),
          "'key' parameter of env_entry must be a nonempty string.");
    }
    if (value == null || value.isEmpty()) {
      throw new EvalException(
          envEntryStruct.getCreationLoc(),
          "'value' parameter of env_entry must be a nonempty string.");
    }
    StringValueParser parser = new StringValueParser(value);
    return new EnvEntry(key, parser.getChunks());
  }

  /** Creates a {@link WithFeatureSet} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static WithFeatureSet withFeatureSetFromSkylark(SkylarkInfo withFeatureSetStruct)
      throws EvalException {
    checkRightProviderType(withFeatureSetStruct, "with_feature_set");
    ImmutableSet<String> features =
        getStringSetFromSkylarkProviderField(withFeatureSetStruct, "features");
    ImmutableSet<String> notFeatures =
        getStringSetFromSkylarkProviderField(withFeatureSetStruct, "not_features");
    return new WithFeatureSet(features, notFeatures);
  }

  /** Creates an {@link EnvSet} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static EnvSet envSetFromSkylark(SkylarkInfo envSetStruct)
      throws InvalidConfigurationException, EvalException {
    checkRightProviderType(envSetStruct, "env_set");
    ImmutableSet<String> actions = getStringSetFromSkylarkProviderField(envSetStruct, "actions");
    if (actions.isEmpty()) {
      throw new EvalException(
          envSetStruct.getCreationLoc(), "actions parameter of env_set must be a nonempty list.");
    }
    ImmutableList.Builder<EnvEntry> envEntryBuilder = ImmutableList.builder();
    ImmutableList<SkylarkInfo> envEntryStructs =
        getSkylarkProviderListFromSkylarkField(envSetStruct, "env_entries");
    for (SkylarkInfo envEntryStruct : envEntryStructs) {
      envEntryBuilder.add(envEntryFromSkylark(envEntryStruct));
    }

    ImmutableSet.Builder<WithFeatureSet> withFeatureSetBuilder = ImmutableSet.builder();
    ImmutableList<SkylarkInfo> withFeatureSetStructs =
        getSkylarkProviderListFromSkylarkField(envSetStruct, "with_features");
    for (SkylarkInfo withFeatureSetStruct : withFeatureSetStructs) {
      withFeatureSetBuilder.add(withFeatureSetFromSkylark(withFeatureSetStruct));
    }
    return new EnvSet(actions, envEntryBuilder.build(), withFeatureSetBuilder.build());
  }

  /** Creates a {@link FlagGroup} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static FlagGroup flagGroupFromSkylark(SkylarkInfo flagGroupStruct)
      throws InvalidConfigurationException, EvalException {
    checkRightProviderType(flagGroupStruct, "flag_group");

    ImmutableList.Builder<Expandable> expandableBuilder = ImmutableList.builder();
    ImmutableList<String> flags = getStringListFromSkylarkProviderField(flagGroupStruct, "flags");
    for (String flag : flags) {
      StringValueParser parser = new StringValueParser(flag);
      expandableBuilder.add(new Flag(parser.getChunks()));
    }

    ImmutableList<SkylarkInfo> flagGroups =
        getSkylarkProviderListFromSkylarkField(flagGroupStruct, "flag_groups");
    for (SkylarkInfo flagGroup : flagGroups) {
      expandableBuilder.add(flagGroupFromSkylark(flagGroup));
    }

    if (flagGroups.size() > 0 && flags.size() > 0) {
      throw new EvalException(
          flagGroupStruct.getCreationLoc(),
          "flag_group must contain either a list of flags or a list of flag_groups.");
    }

    if (flagGroups.size() == 0 && flags.size() == 0) {
      throw new EvalException(
          flagGroupStruct.getCreationLoc(), "Both 'flags' and 'flag_groups' are empty.");
    }

    String iterateOver = getFieldFromSkylarkProvider(flagGroupStruct, "iterate_over", String.class);
    String expandIfAvailable =
        getFieldFromSkylarkProvider(flagGroupStruct, "expand_if_available", String.class);
    String expandIfNotAvailable =
        getFieldFromSkylarkProvider(flagGroupStruct, "expand_if_not_available", String.class);
    String expandIfTrue =
        getFieldFromSkylarkProvider(flagGroupStruct, "expand_if_true", String.class);
    String expandIfFalse =
        getFieldFromSkylarkProvider(flagGroupStruct, "expand_if_false", String.class);
    SkylarkInfo expandIfEqualStruct =
        getFieldFromSkylarkProvider(flagGroupStruct, "expand_if_equal", SkylarkInfo.class);
    VariableWithValue expandIfEqual =
        expandIfEqualStruct == null ? null : variableWithValueFromSkylark(expandIfEqualStruct);

    return new FlagGroup(
        expandableBuilder.build(),
        iterateOver,
        expandIfAvailable == null ? ImmutableSet.of() : ImmutableSet.of(expandIfAvailable),
        expandIfNotAvailable == null ? ImmutableSet.of() : ImmutableSet.of(expandIfNotAvailable),
        expandIfTrue,
        expandIfFalse,
        expandIfEqual);
  }

  /** Creates a {@link FlagSet} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static FlagSet flagSetFromSkylark(SkylarkInfo flagSetStruct)
      throws InvalidConfigurationException, EvalException {
    checkRightProviderType(flagSetStruct, "flag_set");
    ImmutableSet<String> actions = getStringSetFromSkylarkProviderField(flagSetStruct, "actions");
    if (actions.isEmpty()) {
      throw new EvalException(
          flagSetStruct.getCreationLoc(), "'actions' field of flag_set must be a nonempty list.");
    }
    ImmutableList.Builder<FlagGroup> flagGroupsBuilder = ImmutableList.builder();
    ImmutableList<SkylarkInfo> flagGroups =
        getSkylarkProviderListFromSkylarkField(flagSetStruct, "flag_groups");
    for (SkylarkInfo flagGroup : flagGroups) {
      flagGroupsBuilder.add(flagGroupFromSkylark(flagGroup));
    }

    ImmutableSet.Builder<WithFeatureSet> withFeatureSetBuilder = ImmutableSet.builder();
    ImmutableList<SkylarkInfo> withFeatureSetStructs =
        getSkylarkProviderListFromSkylarkField(flagSetStruct, "with_features");
    for (SkylarkInfo withFeatureSetStruct : withFeatureSetStructs) {
      withFeatureSetBuilder.add(withFeatureSetFromSkylark(withFeatureSetStruct));
    }

    return new FlagSet(
        actions, ImmutableSet.of(), withFeatureSetBuilder.build(), flagGroupsBuilder.build());
  }

  /**
   * Creates a {@link com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Tool} from a
   * {@link SkylarkInfo}.
   */
  @VisibleForTesting
  static CcToolchainFeatures.Tool toolFromSkylark(SkylarkInfo toolStruct) throws EvalException {
    checkRightProviderType(toolStruct, "tool");
    String toolPathString = getFieldFromSkylarkProvider(toolStruct, "path", String.class);
    PathFragment toolPath = toolPathString == null ? null : PathFragment.create(toolPathString);
    if (toolPath != null && toolPath.isEmpty()) {
      throw new EvalException(
          toolStruct.getCreationLoc(), "The 'path' field of tool must be a nonempty string.");
    }
    ImmutableSet.Builder<WithFeatureSet> withFeatureSetBuilder = ImmutableSet.builder();
    ImmutableList<SkylarkInfo> withFeatureSetStructs =
        getSkylarkProviderListFromSkylarkField(toolStruct, "with_features");
    for (SkylarkInfo withFeatureSetStruct : withFeatureSetStructs) {
      withFeatureSetBuilder.add(withFeatureSetFromSkylark(withFeatureSetStruct));
    }

    ImmutableSet<String> executionRequirements =
        getStringSetFromSkylarkProviderField(toolStruct, "execution_requirements");
    return new CcToolchainFeatures.Tool(
        toolPath, executionRequirements, withFeatureSetBuilder.build());
  }

  /** Creates an {@link ActionConfig} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static ActionConfig actionConfigFromSkylark(SkylarkInfo actionConfigStruct)
      throws InvalidConfigurationException, EvalException {
    checkRightProviderType(actionConfigStruct, "action_config");
    String actionName =
        getFieldFromSkylarkProvider(actionConfigStruct, "action_name", String.class);
    if (actionName == null || actionName.isEmpty()) {
      throw new EvalException(
          actionConfigStruct.getCreationLoc(),
          "The 'action_name' field of action_config must be a nonempty string.");
    }
    if (!actionName.matches("^[_a-z+\\-]*$")) {
      throw new EvalException(
          actionConfigStruct.getCreationLoc(),
          String.format(
              "An action_config's name must consist solely of lowercase ASCII letters, '_', '+', "
                  + "and '-', got '%s'",
              actionName));
    }

    Boolean enabled = getFieldFromSkylarkProvider(actionConfigStruct, "enabled", Boolean.class);

    ImmutableList.Builder<CcToolchainFeatures.Tool> toolBuilder = ImmutableList.builder();
    ImmutableList<SkylarkInfo> toolStructs =
        getSkylarkProviderListFromSkylarkField(actionConfigStruct, "tools");
    for (SkylarkInfo toolStruct : toolStructs) {
      toolBuilder.add(toolFromSkylark(toolStruct));
    }

    ImmutableList.Builder<FlagSet> flagSetBuilder = ImmutableList.builder();
    ImmutableList<SkylarkInfo> flagSets =
        getSkylarkProviderListFromSkylarkField(actionConfigStruct, "flag_sets");
    for (SkylarkInfo flagSet : flagSets) {
      flagSetBuilder.add(flagSetFromSkylark(flagSet));
    }

    ImmutableList<String> implies =
        getStringListFromSkylarkProviderField(actionConfigStruct, "implies");

    return new ActionConfig(
        actionName, actionName, toolBuilder.build(), flagSetBuilder.build(), enabled, implies);
  }

  /** Creates an {@link ArtifactNamePattern} from a {@link SkylarkInfo}. */
  @VisibleForTesting
  static ArtifactNamePattern artifactNamePatternFromSkylark(SkylarkInfo artifactNamePatternStruct)
      throws EvalException {
    checkRightProviderType(artifactNamePatternStruct, "artifact_name_pattern");
    String categoryName =
        getFieldFromSkylarkProvider(artifactNamePatternStruct, "category_name", String.class);
    if (categoryName == null || categoryName.isEmpty()) {
      throw new EvalException(
          artifactNamePatternStruct.getCreationLoc(),
          "The 'category_name' field of artifact_name_pattern must be a nonempty string.");
    }
    ArtifactCategory foundCategory = null;
    for (ArtifactCategory artifactCategory : ArtifactCategory.values()) {
      if (categoryName.equals(artifactCategory.getCategoryName())) {
        foundCategory = artifactCategory;
      }
    }

    if (foundCategory == null) {
      throw new EvalException(
          artifactNamePatternStruct.getCreationLoc(),
          String.format("Artifact category %s not recognized.", categoryName));
    }

    String extension =
        Strings.nullToEmpty(
            getFieldFromSkylarkProvider(artifactNamePatternStruct, "extension", String.class));
    if (!foundCategory.getAllowedExtensions().contains(extension)) {
      throw new EvalException(
          artifactNamePatternStruct.getCreationLoc(),
          String.format(
              "Unrecognized file extension '%s', allowed extensions are %s,"
                  + " please check artifact_name_pattern configuration for %s in your rule.",
              extension,
              StringUtil.joinEnglishList(foundCategory.getAllowedExtensions(), "or", "'"),
              foundCategory.getCategoryName()));
    }

    String prefix =
        Strings.nullToEmpty(
            getFieldFromSkylarkProvider(artifactNamePatternStruct, "prefix", String.class));
    return new ArtifactNamePattern(foundCategory, prefix, extension);
  }

  private static <T> T getFieldFromSkylarkProvider(
      SkylarkInfo provider, String fieldName, Class<T> clazz) throws EvalException {
    Object obj = provider.getValueOrNull(fieldName);
    if (obj == null) {
      throw new EvalException(
          provider.getCreationLoc(), String.format("Missing mandatory field '%s'", fieldName));
    }
    if (clazz.isInstance(obj)) {
      return clazz.cast(obj);
    }
    if (NoneType.class.isInstance(obj)) {
      return null;
    }
    throw new EvalException(
        provider.getCreationLoc(),
        String.format("Field '%s' is not of '%s' type.", fieldName, clazz.getName()));
  }

  /** Returns a list of strings from a field of a {@link SkylarkInfo}. */
  private static ImmutableList<String> getStringListFromSkylarkProviderField(
      SkylarkInfo provider, String fieldName) throws EvalException {
    return SkylarkList.castSkylarkListOrNoneToList(
            provider.getValueOrNull(fieldName), String.class, fieldName)
        .stream()
        .collect(ImmutableList.toImmutableList());
  }

  /** Returns a set of strings from a field of a {@link SkylarkInfo}. */
  private static ImmutableSet<String> getStringSetFromSkylarkProviderField(
      SkylarkInfo provider, String fieldName) throws EvalException {
    return SkylarkList.castSkylarkListOrNoneToList(
            provider.getValueOrNull(fieldName), String.class, fieldName)
        .stream()
        .collect(ImmutableSet.toImmutableSet());
  }

  /** Returns a list of SkylarkInfo providers from a field of a {@link SkylarkInfo}. */
  private static ImmutableList<SkylarkInfo> getSkylarkProviderListFromSkylarkField(
      SkylarkInfo provider, String fieldName) throws EvalException {
    return SkylarkList.castSkylarkListOrNoneToList(
            provider.getValueOrNull(fieldName), SkylarkInfo.class, fieldName)
        .stream()
        .collect(ImmutableList.toImmutableList());
  }

  private static ImmutableMap<CompilationMode, ImmutableList<String>>
      getCompilationModeFlagsFromSkylark(Object compilationModeFlags, String field)
          throws EvalException {
    Map<String, SkylarkList> compilationModeLinkerFlagsMap =
        SkylarkDict.castSkylarkDictOrNoneToDict(
            compilationModeFlags, String.class, SkylarkList.class, field);
    ImmutableMap.Builder<CompilationMode, ImmutableList<String>> compilationModeFlagsBuilder =
        ImmutableMap.builder();
    for (Entry<String, SkylarkList> entry : compilationModeLinkerFlagsMap.entrySet()) {
      compilationModeFlagsBuilder.put(
          CompilationMode.valueOf(entry.getKey()),
          ImmutableList.copyOf(
              convertSkylarkListOrNestedSetToList(entry.getValue(), String.class)));
    }
    return compilationModeFlagsBuilder.build();
  }

  private static void getLegacyArtifactNamePatterns(
      ImmutableList.Builder<ArtifactNamePattern> patterns) {
    Set<ArtifactCategory> definedCategories = new HashSet<>();
    for (ArtifactNamePattern pattern : patterns.build()) {
      try {
        definedCategories.add(
            ArtifactCategory.valueOf(
                pattern.getArtifactCategory().getCategoryName().toUpperCase(Locale.ENGLISH)));
      } catch (IllegalArgumentException e) {
        // Invalid category name, will be detected later.
        continue;
      }
    }

    for (ArtifactCategory category : ArtifactCategory.values()) {
      if (!definedCategories.contains(category)
          && category.getDefaultPrefix() != null
          && category.getDefaultExtension() != null) {
        patterns.add(
            new ArtifactNamePattern(
                category, category.getDefaultPrefix(), category.getDefaultExtension()));
      }
    }
  }
}
