// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.objc.AppleLinkingOutputs.TargetTriplet;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;

/** Support utility for creating multi-arch Apple binaries. */
public class MultiArchBinarySupport {
  private MultiArchBinarySupport() {}

  private static HashSet<PathFragment> buildAvoidLibrarySet(
      Iterable<CcLinkingContext> avoidDepContexts) {
    HashSet<PathFragment> avoidLibrarySet = new HashSet<>();
    for (CcLinkingContext context : avoidDepContexts) {
      for (LinkerInput linkerInput : context.getLinkerInputs().toList()) {
        for (LibraryToLink libraryToLink : linkerInput.getLibraries()) {
          Artifact library = CompilationSupport.getStaticLibraryForLinking(libraryToLink);
          if (library != null) {
            avoidLibrarySet.add(library.getRunfilesPath());
          }
        }
      }
    }

    return avoidLibrarySet;
  }

  public static CcLinkingContext ccLinkingContextSubtractSubtrees(
      RuleContext ruleContext,
      Iterable<CcLinkingContext> depContexts,
      Iterable<CcLinkingContext> avoidDepContexts) {
    CcLinkingContext.Builder outputContext = new CcLinkingContext.Builder();

    outputContext.setOwner(ruleContext.getLabel());

    HashSet<PathFragment> avoidLibrarySet = buildAvoidLibrarySet(avoidDepContexts);
    ImmutableList.Builder<LibraryToLink> filteredLibraryList = new ImmutableList.Builder<>();
    for (CcLinkingContext context : depContexts) {

      for (LinkerInput linkerInput : context.getLinkerInputs().toList()) {
        for (LibraryToLink libraryToLink : linkerInput.getLibraries()) {
          Artifact library = CompilationSupport.getLibraryForLinking(libraryToLink);
          Preconditions.checkNotNull(library);
          if (!avoidLibrarySet.contains(library.getRunfilesPath())) {
            filteredLibraryList.add(libraryToLink);
          }
        }

        outputContext.addUserLinkFlags(linkerInput.getUserLinkFlags());
        outputContext.addNonCodeInputs(linkerInput.getNonCodeInputs());
        outputContext.addLinkstamps(linkerInput.getLinkstamps());
      }

      NestedSetBuilder<LibraryToLink> filteredLibrarySet = NestedSetBuilder.linkOrder();
      filteredLibrarySet.addAll(filteredLibraryList.build());
      outputContext.addLibraries(filteredLibrarySet.build().toList());
    }

    return outputContext.build();
  }

  /**
   * Returns an Apple target triplet (arch, platform, environment) for a given {@link
   * BuildConfigurationValue}.
   *
   * @param config {@link BuildConfigurationValue} from rule context
   * @return {@link AppleLinkingOutputs.TargetTriplet}
   */
  private static AppleLinkingOutputs.TargetTriplet getTargetTriplet(
      BuildConfigurationValue config) {
    // TODO(b/177442911): Use the target platform from platform info coming from split
    // transition outputs instead of inferring this based on the target CPU.
    ApplePlatform cpuPlatform = ApplePlatform.forTargetCpu(config.getCpu());
    AppleConfiguration appleConfig = config.getFragment(AppleConfiguration.class);

    return TargetTriplet.create(
        appleConfig.getSingleArchitecture(),
        cpuPlatform.getTargetPlatform(),
        cpuPlatform.getTargetEnvironment());
  }

  /**
   * Transforms a {@link Map<Optional<String>, List<ConfiguredTargetAndData>>}, to a Starlark Dict
   * keyed by split transition keys with {@link AppleLinkingOutputs.TargetTriplet} Starlark struct
   * definition.
   *
   * @param ctads a {@link Map<Optional<String>, List<ConfiguredTargetAndData>>} from rule context
   * @return a Starlark {@link Dict<String, StructImpl>} representing split transition keys with
   *     their target triplet (architecture, platform, environment)
   */
  public static Dict<String, StructImpl> getSplitTargetTripletFromCtads(
      Map<Optional<String>, List<ConfiguredTargetAndData>> ctads) throws EvalException {
    Dict.Builder<String, StructImpl> result = Dict.builder();
    for (Optional<String> splitTransitionKey : ctads.keySet()) {
      if (!splitTransitionKey.isPresent()) {
        throw new EvalException("unexpected empty key in split transition");
      }
      TargetTriplet targetTriplet =
          getTargetTriplet(
              Iterables.getOnlyElement(ctads.get(splitTransitionKey)).getConfiguration());
      result.put(splitTransitionKey.get(), targetTriplet.toStarlarkStruct());
    }
    return result.buildImmutable();
  }
}
