// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.skylark.BazelStarlarkContext;
import com.google.devtools.build.lib.analysis.skylark.SkylarkActionFactory;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper;
import com.google.devtools.build.lib.rules.cpp.CcLinkingOutputs;
import com.google.devtools.build.lib.rules.cpp.CcModule;
import com.google.devtools.build.lib.rules.cpp.CcToolchainConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.FdoContext;
import com.google.devtools.build.lib.rules.cpp.FeatureConfigurationForStarlark;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.BazelCcModuleApi;
import com.google.devtools.build.lib.skylarkinterface.StarlarkContext;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import java.util.stream.Stream;

/**
 * A module that contains Skylark utilities for C++ support.
 *
 * <p>This is a work in progress. The API is guarded behind
 * --experimental_cc_skylark_api_enabled_packages. The API is under development and unstable.
 */
public class BazelCcModule extends CcModule
    implements BazelCcModuleApi<
        SkylarkActionFactory,
        Artifact,
        SkylarkRuleContext,
        CcToolchainProvider,
        FeatureConfigurationForStarlark,
        CcCompilationContext,
        CcCompilationOutputs,
        CcLinkingOutputs,
        LibraryToLink,
        CcLinkingContext,
        CcToolchainVariables,
        CcToolchainConfigInfo> {
  private static final ImmutableList<String> SUPPORTED_OUTPUT_TYPES =
      ImmutableList.of("executable", "dynamic_library");

  private enum Language {
    CPP("c++"),
    OBJC("objc"),
    OBJCPP("objc++");

    private final String representation;

    Language(String representation) {
      this.representation = representation;
    }

    public String getRepresentation() {
      return representation;
    }
  }

  @Override
  public CppSemantics getSemantics() {
    return BazelCppSemantics.INSTANCE;
  }

  @Override
  public Tuple<Object> compile(
      SkylarkActionFactory skylarkActionFactoryApi,
      FeatureConfigurationForStarlark skylarkFeatureConfiguration,
      CcToolchainProvider skylarkCcToolchainProvider,
      SkylarkList<Artifact> sources,
      SkylarkList<Artifact> publicHeaders,
      SkylarkList<Artifact> privateHeaders,
      SkylarkList<String> includes,
      SkylarkList<String> quoteIncludes,
      SkylarkList<String> systemIncludes,
      SkylarkList<String> defines,
      SkylarkList<String> userCompileFlags,
      SkylarkList<CcCompilationContext> ccCompilationContexts,
      String name,
      boolean disallowPicOutputs,
      boolean disallowNopicOutputs,
      Location location,
      Environment environment)
      throws EvalException {
    return compile(
        skylarkActionFactoryApi,
        skylarkFeatureConfiguration,
        skylarkCcToolchainProvider,
        sources,
        publicHeaders,
        privateHeaders,
        includes,
        quoteIncludes,
        systemIncludes,
        defines,
        userCompileFlags,
        ccCompilationContexts,
        name,
        disallowPicOutputs,
        disallowNopicOutputs,
        /* grepIncludes= */ null,
        SkylarkList.createImmutable(ImmutableList.of()),
        location,
        environment);
  }

  @Override
  public Tuple<Object> createLinkingContextFromCompilationOutputs(
      SkylarkActionFactory skylarkActionFactoryApi,
      FeatureConfigurationForStarlark skylarkFeatureConfiguration,
      CcToolchainProvider skylarkCcToolchainProvider,
      CcCompilationOutputs compilationOutputs,
      SkylarkList<String> userLinkFlags,
      SkylarkList<CcLinkingContext> linkingContexts,
      String name,
      String language,
      boolean alwayslink,
      SkylarkList<Artifact> additionalInputs,
      boolean disallowStaticLibraries,
      boolean disallowDynamicLibraries,
      Location location,
      Environment environment,
      StarlarkContext starlarkContext)
      throws InterruptedException, EvalException {
    CcCommon.checkLocationWhitelisted(
        environment.getSemantics(),
        location,
        environment.getGlobals().getLabel().getPackageIdentifier().toString());
    validateLanguage(location, language);
    SkylarkActionFactory actions = skylarkActionFactoryApi;
    CcToolchainProvider ccToolchainProvider = convertFromNoneable(skylarkCcToolchainProvider, null);
    FeatureConfigurationForStarlark featureConfiguration =
        convertFromNoneable(skylarkFeatureConfiguration, null);
    Label label = getCallerLabel(location, environment, name);
    FdoContext fdoContext = ccToolchainProvider.getFdoContext();
    LinkTargetType staticLinkTargetType = null;
    if (language.equals(Language.CPP.getRepresentation())) {
      staticLinkTargetType = LinkTargetType.STATIC_LIBRARY;
    } else if (language.equals(Language.OBJC.getRepresentation())
        || language.equals(Language.OBJCPP.getRepresentation())) {
      staticLinkTargetType = LinkTargetType.OBJC_ARCHIVE;
    } else {
      throw new IllegalStateException("Language is not valid.");
    }
    CcLinkingHelper helper =
        new CcLinkingHelper(
                actions.getActionConstructionContext().getRuleErrorConsumer(),
                label,
                actions.asActionRegistry(location, actions),
                actions.getActionConstructionContext(),
                BazelCppSemantics.INSTANCE,
                featureConfiguration.getFeatureConfiguration(),
                ccToolchainProvider,
                fdoContext,
                actions.getActionConstructionContext().getConfiguration(),
                actions
                    .getActionConstructionContext()
                    .getConfiguration()
                    .getFragment(CppConfiguration.class),
                ((BazelStarlarkContext) starlarkContext).getSymbolGenerator())
            .addNonCodeLinkerInputs(additionalInputs)
            .setShouldCreateStaticLibraries(!disallowStaticLibraries)
            .setShouldCreateDynamicLibrary(
                !disallowDynamicLibraries
                    && !featureConfiguration
                        .getFeatureConfiguration()
                        .isEnabled(CppRuleClasses.TARGETS_WINDOWS))
            .setStaticLinkType(staticLinkTargetType)
            .setDynamicLinkType(LinkTargetType.NODEPS_DYNAMIC_LIBRARY)
            .addLinkopts(userLinkFlags);
    try {
      CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
      ImmutableList<LibraryToLink> libraryToLink = ImmutableList.of();
      if (!compilationOutputs.isEmpty()) {
        ccLinkingOutputs = helper.link(compilationOutputs);
        if (!ccLinkingOutputs.isEmpty()) {
          libraryToLink =
              ImmutableList.of(
                  ccLinkingOutputs.getLibraryToLink().toBuilder()
                      .setAlwayslink(alwayslink)
                      .build());
        }
      }
      CcLinkingContext linkingContext =
          helper.buildCcLinkingContextFromLibrariesToLink(
              libraryToLink, CcCompilationContext.EMPTY);
      return Tuple.of(
          CcLinkingContext.merge(
              ImmutableList.<CcLinkingContext>builder()
                  .add(linkingContext)
                  .addAll(linkingContexts)
                  .build()),
          ccLinkingOutputs);
    } catch (RuleErrorException e) {
      throw new EvalException(location, e);
    }
  }

  @Override
  public CcLinkingOutputs link(
      SkylarkActionFactory actions,
      FeatureConfigurationForStarlark skylarkFeatureConfiguration,
      CcToolchainProvider skylarkCcToolchainProvider,
      CcCompilationOutputs compilationOutputs,
      SkylarkList<String> userLinkFlags,
      SkylarkList<CcLinkingContext> linkingContexts,
      String name,
      String language,
      String outputType,
      boolean linkDepsStatically,
      SkylarkList<Artifact> additionalInputs,
      Location location,
      Environment environment,
      StarlarkContext starlarkContext)
      throws InterruptedException, EvalException {
    CcCommon.checkLocationWhitelisted(
        environment.getSemantics(),
        location,
        environment.getGlobals().getLabel().getPackageIdentifier().toString());
    validateLanguage(location, language);
    validateOutputType(location, outputType);
    CcToolchainProvider ccToolchainProvider = convertFromNoneable(skylarkCcToolchainProvider, null);
    FeatureConfigurationForStarlark featureConfiguration =
        convertFromNoneable(skylarkFeatureConfiguration, null);
    Label label = getCallerLabel(location, environment, name);
    FdoContext fdoContext = ccToolchainProvider.getFdoContext();
    LinkTargetType dynamicLinkTargetType = null;
    if (language.equals(Language.CPP.getRepresentation())) {
      if (outputType.equals("executable")) {
        dynamicLinkTargetType = LinkTargetType.EXECUTABLE;
      } else if (outputType.equals("dynamic_library")) {
        dynamicLinkTargetType = LinkTargetType.DYNAMIC_LIBRARY;
      }
    } else if (language.equals(Language.OBJC.getRepresentation())
        && outputType.equals("executable")) {
      dynamicLinkTargetType = LinkTargetType.OBJC_EXECUTABLE;
    } else if (language.equals(Language.OBJCPP.getRepresentation())
        && outputType.equals("executable")) {
      dynamicLinkTargetType = LinkTargetType.OBJCPP_EXECUTABLE;
    } else {
      throw new EvalException(
          location, "Language '" + language + "' does not support " + outputType);
    }

    CcLinkingHelper helper =
        new CcLinkingHelper(
                actions.getActionConstructionContext().getRuleErrorConsumer(),
                label,
                actions.asActionRegistry(location, actions),
                actions.getActionConstructionContext(),
                BazelCppSemantics.INSTANCE,
                featureConfiguration.getFeatureConfiguration(),
                ccToolchainProvider,
                fdoContext,
                actions.getActionConstructionContext().getConfiguration(),
                actions
                    .getActionConstructionContext()
                    .getConfiguration()
                    .getFragment(CppConfiguration.class),
                ((BazelStarlarkContext) starlarkContext).getSymbolGenerator())
            .setLinkingMode(linkDepsStatically ? LinkingMode.STATIC : LinkingMode.DYNAMIC)
            .addNonCodeLinkerInputs(additionalInputs)
            .setDynamicLinkType(dynamicLinkTargetType)
            .addCcLinkingContexts(linkingContexts)
            .addLinkopts(userLinkFlags);
    try {
      CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
      if (!compilationOutputs.isEmpty()) {
        ccLinkingOutputs = helper.link(compilationOutputs);
      }
      return ccLinkingOutputs;
    } catch (RuleErrorException e) {
      throw new EvalException(location, e);
    }
  }

  private void validateLanguage(Location location, String language) throws EvalException {
    if (!Stream.of(Language.values())
        .map(Language::getRepresentation)
        .collect(ImmutableList.toImmutableList())
        .contains(language)) {
      throw new EvalException(location, "Language '" + language + "' is not supported");
    }
  }

  private void validateOutputType(Location location, String outputType) throws EvalException {
    if (!SUPPORTED_OUTPUT_TYPES.contains(outputType)) {
      throw new EvalException(location, "Output type '" + outputType + "' is not supported");
    }
  }
}
