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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.rules.cpp.CcModule.CcSkylarkInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcModuleApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcSkylarkInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
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
        isStaticLinkingMode);
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
}
