// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * A class to create C/C++ compile actions in a way that is consistent with cc_library. Rules that
 * generate source files and emulate cc_library on top of that should use this class instead of the
 * lower-level APIs in CppHelper and CppCompileActionBuilder.
 *
 * <p>Rules that want to use this class are required to have implicit dependencies on the toolchain,
 * the STL, and so on. Optionally, they can also have copts, and malloc attributes, but note that
 * these require explicit calls to the corresponding setter methods.
 */
public final class CcStaticCompilationHelper {

  private CcStaticCompilationHelper() {}

  // Methods creating actions:

  static void createCompileActionTemplate(
      ActionConstructionContext actionConstructionContext,
      CcToolchainProvider ccToolchain,
      BuildConfigurationValue configuration,
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables compileBuildVariables,
      CppSemantics semantics,
      CppSource source,
      CppCompileActionBuilder builder,
      ImmutableList<ArtifactCategory> outputCategories,
      SpecialArtifact outputFiles,
      SpecialArtifact dotdTreeArtifact,
      SpecialArtifact diagnosticsTreeArtifact,
      @Nullable SpecialArtifact ltoIndexTreeArtifact)
      throws RuleErrorException, EvalException {
    SpecialArtifact sourceArtifact = (SpecialArtifact) source.getSource();
    builder.setVariables(compileBuildVariables);
    semantics.finalizeCompileActionBuilder(configuration, featureConfiguration, builder);

    ActionOwner actionOwner = null;
    if (actionConstructionContext instanceof RuleContext ruleContext
        && ruleContext.useAutoExecGroups()) {
      actionOwner =
          actionConstructionContext.getActionOwner(semantics.getCppToolchainType().toString());
    }
    try {
      CppCompileActionTemplate actionTemplate =
          new CppCompileActionTemplate(
              sourceArtifact,
              outputFiles,
              dotdTreeArtifact,
              diagnosticsTreeArtifact,
              ltoIndexTreeArtifact,
              builder,
              ccToolchain,
              outputCategories,
              actionOwner == null ? actionConstructionContext.getActionOwner() : actionOwner);
      actionConstructionContext.registerAction(actionTemplate);
    } catch (EvalException e) {
      throw new RuleErrorException(e.getMessage());
    }
  }

  // Helper methods used when creating actions:

  static ImmutableList<String> collectPerFileCopts(
      CppConfiguration cppConfiguration, Artifact sourceFile, Label sourceLabel) {
    return cppConfiguration.getPerFileCopts().stream()
        .filter(
            perLabelOptions ->
                (sourceLabel != null && perLabelOptions.isIncluded(sourceLabel))
                    || perLabelOptions.isIncluded(sourceFile))
        .map(PerLabelOptions::getOptions)
        .flatMap(options -> options.stream())
        .collect(ImmutableList.toImmutableList());
  }

  /**
   * Return flags that were specified on the Blaze command line. Take the filetype of sourceFilename
   * into account.
   */
  public static ImmutableList<String> getCoptsFromOptions(
      CppConfiguration config, CppSemantics semantics, String sourceFilename) {
    ImmutableList.Builder<String> flagsBuilder = ImmutableList.builder();

    flagsBuilder.addAll(config.getCopts());

    if (CppFileTypes.C_SOURCE.matches(sourceFilename)) {
      flagsBuilder.addAll(config.getConlyopts());
    }

    if (CppFileTypes.CPP_SOURCE.matches(sourceFilename)
        || CppFileTypes.CPP_HEADER.matches(sourceFilename)
        || CppFileTypes.CPP_MODULE_MAP.matches(sourceFilename)
        || CppFileTypes.OBJCPP_SOURCE.matches(sourceFilename)
        || CppFileTypes.CLIF_INPUT_PROTO.matches(sourceFilename)) {
      flagsBuilder.addAll(config.getCxxopts());
    }

    if (CppFileTypes.OBJC_SOURCE.matches(sourceFilename)
        || CppFileTypes.OBJCPP_SOURCE.matches(sourceFilename)
        || (CppFileTypes.CPP_HEADER.matches(sourceFilename)
            && semantics.language() == Language.OBJC)) {
      flagsBuilder.addAll(config.getObjcopts());
    }

    return flagsBuilder.build();
  }
}
