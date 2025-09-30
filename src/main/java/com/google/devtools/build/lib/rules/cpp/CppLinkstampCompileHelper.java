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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppLinkActionBuilder.LinkActionConstruction;
import net.starlark.java.eval.EvalException;

/** Handles creation of CppCompileAction used to compile linkstamp sources. */
public final class CppLinkstampCompileHelper {

  /** Creates {@link CppCompileAction} to compile linkstamp source. */
  public static CppCompileAction createLinkstampCompileAction(
      LinkActionConstruction linkActionConstruction,
      Artifact sourceFile,
      Artifact outputFile,
      NestedSet<Artifact> compilationInputs,
      NestedSet<Artifact> nonCodeInputs,
      NestedSet<Artifact> inputsForInvalidation,
      ImmutableList<Artifact> buildInfoHeaderArtifacts,
      CcToolchainProvider ccToolchainProvider,
      FeatureConfiguration featureConfiguration,
      CppSemantics semantics,
      CcToolchainVariables compileBuildVariables)
      throws EvalException {
    CppCompileActionBuilder builder =
        new CppCompileActionBuilder(
                linkActionConstruction.getContext(),
                ccToolchainProvider,
                linkActionConstruction.getConfig(),
                semantics)
            .addMandatoryInputs(compilationInputs)
            .setVariables(compileBuildVariables)
            .setFeatureConfiguration(featureConfiguration)
            .setSourceFile(sourceFile)
            .setOutputs(outputFile, /* dotdFile= */ null, /* diagnosticsFile= */ null)
            .setCacheKeyInputs(inputsForInvalidation)
            .setBuildInfoHeaderArtifacts(buildInfoHeaderArtifacts)
            .addMandatoryInputs(nonCodeInputs)
            .setShareable(true)
            .setShouldScanIncludes(false)
            .setActionName(CppActionNames.LINKSTAMP_COMPILE);

    semantics.finalizeCompileActionBuilder(
        linkActionConstruction.getConfig(), featureConfiguration, builder);
    return builder.buildOrThrowIllegalStateException();
  }

  private CppLinkstampCompileHelper() {}
}
