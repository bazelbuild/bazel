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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.skylark.BazelStarlarkContext;
import com.google.devtools.build.lib.analysis.skylark.SkylarkActionFactory;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcCompilationOutputs;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper.LinkingInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingOutputs;
import com.google.devtools.build.lib.rules.cpp.CcModule;
import com.google.devtools.build.lib.rules.cpp.CcToolchainConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.FdoContext;
import com.google.devtools.build.lib.rules.cpp.FeatureConfigurationForStarlark;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink.CcLinkingContext;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.BazelCcModuleApi;
import com.google.devtools.build.lib.skylarkinterface.StarlarkContext;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/**
 * A module that contains Skylark utilities for C++ support.
 *
 * <p>This is a work in progress. The API is guarded behind
 * --experimental_cc_skylark_api_enabled_packages. The API is under development and unstable.
 */
public class BazelCcModule extends CcModule
    implements BazelCcModuleApi<
    Artifact,
    SkylarkRuleContext,
    SkylarkActionFactory,
    CcToolchainProvider,
    FeatureConfigurationForStarlark,
    CcCompilationContext,
    CcCompilationOutputs,
    CcLinkingContext,
    LibraryToLink,
    CcToolchainVariables,
    CcToolchainConfigInfo> {
  public static final FileTypeSet ALL_C_CLASS_SOURCE =
      FileTypeSet.of(
          CppFileTypes.CPP_SOURCE,
          CppFileTypes.C_SOURCE,
          CppFileTypes.OBJCPP_SOURCE,
          CppFileTypes.OBJC_SOURCE);

  @Override
  public Tuple<Object> compile(
      SkylarkActionFactory actions,
      FeatureConfigurationForStarlark skylarkFeatureConfiguration,
      CcToolchainProvider skylarkCcToolchainProvider,
      SkylarkList<Artifact> sources,
      SkylarkList<Artifact> publicHeaders,
      SkylarkList<Artifact> privateHeaders,
      SkylarkList<String> skylarkIncludes,
      SkylarkList<String> skylarkUserCompileFlags,
      SkylarkList<CcCompilationContext> ccCompilationContexts,
      String name,
      Location location)
      throws EvalException, InterruptedException {
    CcToolchainProvider ccToolchainProvider = convertFromNoneable(skylarkCcToolchainProvider, null);
    FeatureConfigurationForStarlark featureConfiguration =
        convertFromNoneable(skylarkFeatureConfiguration, null);
    Pair<List<Artifact>, List<Artifact>> separatedHeadersAndSources =
        separateSourcesFromHeaders(sources);
    FdoContext fdoContext = ccToolchainProvider.getFdoContext();
    // TODO(plf): Need to flatten the nested set to convert the Strings to PathFragment. This could
    // be avoided if path fragments are ever added to Skylark or in the C++ code we take Strings
    // instead of PathFragments.
    List<String> includeDirs = convertSkylarkListOrNestedSetToList(skylarkIncludes, String.class);

    validateExtensions(
        location,
        "srcs",
        sources,
        ALL_C_CLASS_SOURCE,
        FileTypeSet.of(CppFileTypes.CPP_SOURCE, CppFileTypes.C_SOURCE));
    validateExtensions(
        location,
        "public_hdrs",
        publicHeaders,
        FileTypeSet.of(CppFileTypes.CPP_HEADER),
        FileTypeSet.of(CppFileTypes.CPP_HEADER));
    validateExtensions(
        location,
        "private_hdrs",
        privateHeaders,
        FileTypeSet.of(CppFileTypes.CPP_HEADER),
        FileTypeSet.of(CppFileTypes.CPP_HEADER));

    Label label = getCallerLabel(location, actions, name);
    CcCompilationHelper helper =
        new CcCompilationHelper(
            actions.asActionRegistry(location, actions),
            actions.getActionConstructionContext(),
            label,
            /* grepIncludes= */ null,
            BazelCppSemantics.INSTANCE,
            featureConfiguration.getFeatureConfiguration(),
            ccToolchainProvider,
            fdoContext)
            .addPublicHeaders(publicHeaders)
            .addIncludeDirs(
                includeDirs.stream()
                    .map(PathFragment::create)
                    .collect(ImmutableList.toImmutableList()))
            .addPrivateHeaders(separatedHeadersAndSources.first)
            .addSources(separatedHeadersAndSources.second)
            .addCcCompilationContexts(ccCompilationContexts)
            .setCopts(skylarkUserCompileFlags);

    try {
      CompilationInfo compilationInfo = helper.compile();
      return Tuple.of(
          compilationInfo.getCcCompilationContext(), compilationInfo.getCcCompilationOutputs());
    } catch (RuleErrorException e) {
      throw new EvalException(location, e);
    }
  }

  @Override
  public Tuple<Object> createLinkingContextFromCompilationOutputs(
      SkylarkActionFactory actions,
      FeatureConfigurationForStarlark skylarkFeatureConfiguration,
      CcToolchainProvider skylarkCcToolchainProvider,
      CcCompilationOutputs compilationOutputs,
      SkylarkList<String> skylarkUserLinkFlags,
      SkylarkList<CcLinkingContext> linkingContexts,
      String name,
      Location location,
      StarlarkContext starlarkContext)
      throws InterruptedException, EvalException {
    CcToolchainProvider ccToolchainProvider = convertFromNoneable(skylarkCcToolchainProvider, null);
    FeatureConfigurationForStarlark featureConfiguration =
        convertFromNoneable(skylarkFeatureConfiguration, null);
    Label label = getCallerLabel(location, actions, name);
    FdoContext fdoContext = ccToolchainProvider.getFdoContext();
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
            .addLinkopts(skylarkUserLinkFlags)
            .addCcLinkingContexts(linkingContexts);
    try {
      CcLinkingOutputs ccLinkingOutputs = CcLinkingOutputs.EMPTY;
      ImmutableList<LibraryToLink> libraryToLink = ImmutableList.of();
      if (!compilationOutputs.isEmpty()) {
        ccLinkingOutputs = helper.link(compilationOutputs);
        if (!ccLinkingOutputs.isEmpty()) {
          libraryToLink =
              ImmutableList.of(ccLinkingOutputs.getLibraryToLink());
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

  @SuppressWarnings("unchecked")
  protected static <T> List<T> convertSkylarkListOrNestedSetToList(Object o, Class<T> type) {
    return o instanceof SkylarkNestedSet
        ? ((SkylarkNestedSet) o).getSet(type).toList()
        : ((SkylarkList) o).getImmutableList();
  }

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


  @SuppressWarnings("unchecked")
  protected static <T> NestedSet<T> convertSkylarkListOrNestedSetToNestedSet(
      Object o, Class<T> type) {
    return o instanceof SkylarkNestedSet
        ? ((SkylarkNestedSet) o).getSet(type)
        : NestedSetBuilder.wrap(Order.COMPILE_ORDER, (SkylarkList<T>) o);
  }

  private void validateExtensions(
      Location location,
      String paramName,
      List<Artifact> files,
      FileTypeSet validFileTypeSet,
      FileTypeSet fileTypeForErrorMessage)
      throws EvalException {
    for (Artifact file : files) {
      if (!validFileTypeSet.matches(file.getFilename())) {
        throw new EvalException(
            location,
            String.format(
                "'%s' has wrong extension. The list of possible extensions for '"
                    + paramName
                    + "' are: %s",
                file.getExecPathString(),
                Joiner.on(",").join(fileTypeForErrorMessage.getExtensions())));
      }
    }
  }

  protected Label getCallerLabel(Location location, SkylarkActionFactory actions, String name)
      throws EvalException {
    Label label;
    try {
      label =
          Label.create(
              actions.getActionConstructionContext().getActionOwner().getLabel().getPackageName(),
              name);
    } catch (LabelSyntaxException e) {
      throw new EvalException(location, e);
    }
    return label;
  }
}
