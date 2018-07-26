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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
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
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper.LinkingInfo;
import com.google.devtools.build.lib.rules.cpp.CcModule.CcSkylarkInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcModuleApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcSkylarkInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
   * C++ Skylark rules should have this provider so that native rules can depend on them. This will
   * eventually go away once b/73921130 is fixed.
   */
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
    return featureConfiguration
        .getToolForAction(actionName)
        .getToolPathFragment()
        .getSafePathString();
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
        asStringNestedSet(userCompileFlags),
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
        asStringNestedSet(userLinkFlags),
        /* interfaceLibraryBuilder= */ null,
        /* interfaceLibraryOutput= */ null,
        /* ltoOutputRootPrefix= */ null,
        convertFromNoneable(defFile, /* defaultValue= */ null),
        /* fdoSupport= */ null,
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

  /** Converts an object that can be the either SkylarkNestedSet or None into NestedSet. */
  protected NestedSet<String> asStringNestedSet(Object o) {
    SkylarkNestedSet skylarkNestedSet = convertFromNoneable(o, /* defaultValue= */ null);
    if (skylarkNestedSet != null) {
      return skylarkNestedSet.getSet(String.class);
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  @Override
  public LibraryToLink createLibraryLinkerInput(
      SkylarkRuleContext skylarkRuleContext, Artifact library, String skylarkArtifactCategory)
      throws EvalException {
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
            skylarkRuleContext.getRuleContext(),
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
      throws EvalException {
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
  public CcSkylarkInfo createCcSkylarkInfo(Object skylarkRuleContextObject) throws EvalException {
    SkylarkRuleContext skylarkRuleContext =
        convertFromNoneable(skylarkRuleContextObject, /* defaultValue= */ null);
    if (skylarkRuleContext != null) {
      CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    }
    return new CcSkylarkInfo();
  }

  @SkylarkCallable(
      name = "merge_cc_linking_infos",
      documented = false,
      parameters = {
        @Param(
            name = "cc_linking_infos",
            doc = "cc_linking_infos to be merged.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class)
      })
  public CcLinkingInfo mergeCcLinkingInfos(SkylarkList<CcLinkingInfo> ccLinkingInfos) {
    return CcLinkingInfo.merge(ccLinkingInfos);
  }

  @SkylarkCallable(
      name = "merge_cc_compilation_infos",
      documented = false,
      parameters = {
        @Param(
            name = "cc_compilation_infos",
            doc = "cc_compilation_infos to be merged.",
            positional = false,
            named = true,
            defaultValue = "[]",
            type = SkylarkList.class)
      })
  public CcCompilationInfo mergeCcCompilationInfos(
      SkylarkList<CcCompilationInfo> ccCompilationInfos) {
    return CcCompilationInfo.merge(ccCompilationInfos);
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
      SkylarkList<CcCompilationInfo> ccCompilationInfos,
      Object purpose)
      throws EvalException {
    CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    RuleContext ruleContext = skylarkRuleContext.getRuleContext();
    CcToolchainProvider ccToolchainProvider = convertFromNoneable(skylarkCcToolchainProvider, null);
    FeatureConfiguration featureConfiguration =
        convertFromNoneable(skylarkFeatureConfiguration, null);
    Pair<List<Artifact>, List<Artifact>> separatedHeadersAndSources =
        separateSourcesFromHeaders(sources);
    FdoSupportProvider fdoSupport =
        CppHelper.getFdoSupportUsingDefaultCcToolchainAttribute(ruleContext);
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
                fdoSupport)
            .addPublicHeaders(headers)
            .addIncludeDirs(
                includeDirs
                    .stream()
                    .map(PathFragment::create)
                    .collect(ImmutableList.toImmutableList()))
            .addPrivateHeaders(separatedHeadersAndSources.first)
            .addSources(separatedHeadersAndSources.second)
            .addCcCompilationInfos(ccCompilationInfos)
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
      throws InterruptedException, EvalException {
    CcCommon.checkRuleWhitelisted(skylarkRuleContext);
    RuleContext ruleContext = skylarkRuleContext.getRuleContext();
    CcToolchainProvider ccToolchainProvider = convertFromNoneable(skylarkCcToolchainProvider, null);
    FeatureConfiguration featureConfiguration =
        convertFromNoneable(skylarkFeatureConfiguration, null);
    FdoSupportProvider fdoSupport =
        CppHelper.getFdoSupportUsingDefaultCcToolchainAttribute(ruleContext);
    NestedSet<String> linkopts =
        convertSkylarkListOrNestedSetToNestedSet(skylarkLinkopts, String.class);
    CcLinkingHelper helper =
        new CcLinkingHelper(
                ruleContext,
                cppSemantics,
                featureConfiguration,
                ccToolchainProvider,
                fdoSupport,
                ruleContext.getConfiguration())
            .addLinkopts(linkopts)
            .setShouldCreateStaticLibraries(shouldCreateStaticLibraries)
            .setDynamicLibrary(convertFromNoneable(dynamicLibrary, null))
            .addCcLinkingInfos(skylarkCcLinkingInfos)
            .setNeverLink(neverLink);
    try {
      return helper.link(ccCompilationOutputs, CcCompilationContext.EMPTY);
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
}
