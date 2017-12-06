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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import java.util.regex.Pattern;

/** Handles creation of CppCompileAction used to compile linkstamp sources. */
public class CppLinkstampCompileHelper {

  /** Creates {@link CppCompileAction} to compile linkstamp source */
  public static CppCompileAction createLinkstampCompileAction(
      RuleContext ruleContext,
      Artifact sourceFile,
      Artifact outputFile,
      Iterable<Artifact> compilationInputs,
      ImmutableSet<Artifact> nonCodeInputs,
      ImmutableList<Artifact> buildInfoHeaderArtifacts,
      Iterable<String> additionalLinkstampDefines,
      CcToolchainProvider ccToolchainProvider,
      boolean codeCoverageEnabled,
      CppConfiguration cppConfiguration,
      String fdoBuildStamp,
      FeatureConfiguration featureConfiguration,
      boolean needsPic,
      String labelReplacement,
      String outputReplacement,
      CppSemantics semantics) {
    CppCompileActionBuilder builder =
        new CppCompileActionBuilder(ruleContext, ccToolchainProvider)
            .addMandatoryInputs(compilationInputs)
            .setVariables(
                getVariables(
                    sourceFile,
                    outputFile,
                    labelReplacement,
                    outputReplacement,
                    additionalLinkstampDefines,
                    buildInfoHeaderArtifacts,
                    featureConfiguration,
                    cppConfiguration,
                    ccToolchainProvider,
                    needsPic,
                    fdoBuildStamp,
                    codeCoverageEnabled))
            .setFeatureConfiguration(featureConfiguration)
            .setSourceFile(sourceFile)
            .setSemantics(semantics)
            .setOutputs(outputFile, null)
            .setBuiltinIncludeFiles(buildInfoHeaderArtifacts)
            .addMandatoryInputs(nonCodeInputs)
            .setCppConfiguration(cppConfiguration)
            .setActionName(CppCompileAction.LINKSTAMP_COMPILE);
    semantics.finalizeCompileActionBuilder(ruleContext, builder);
    return builder.buildOrThrowIllegalStateException();
  }

  private static ImmutableList<String> computeAllLinkstampDefines(
      String labelReplacement,
      String outputReplacement,
      Iterable<String> additionalLinkstampDefines,
      CppConfiguration cppConfiguration,
      String fdoBuildStamp,
      boolean codeCoverageEnabled) {
    String labelPattern = Pattern.quote("${LABEL}");
    String outputPathPattern = Pattern.quote("${OUTPUT_PATH}");
    Builder<String> defines =
        ImmutableList.<String>builder()
            .add("GPLATFORM=\"" + cppConfiguration + "\"")
            .add("BUILD_COVERAGE_ENABLED=" + (codeCoverageEnabled ? "1" : "0"))
            .add(CppConfiguration.FDO_STAMP_MACRO + "=\"" + fdoBuildStamp + "\"")
            // G3_VERSION_INFO and G3_TARGET_NAME are C string literals that normally
            // contain the label of the target being linked.  However, they are set
            // differently when using shared native deps. In that case, a single .so file
            // is shared by multiple targets, and its contents cannot depend on which
            // target(s) were specified on the command line.  So in that case we have
            // to use the (obscure) name of the .so file instead, or more precisely
            // the path of the .so file relative to the workspace root.
            .add("G3_VERSION_INFO=\"${LABEL}\"")
            .add("G3_TARGET_NAME=\"${LABEL}\"")
            // G3_BUILD_TARGET is a C string literal containing the output of this
            // link.  (An undocumented and untested invariant is that G3_BUILD_TARGET is the
            // location of the executable, either absolutely, or relative to the directory part of
            // BUILD_INFO.)
            .add("G3_BUILD_TARGET=\"${OUTPUT_PATH}\"")
            .addAll(additionalLinkstampDefines);

    if (fdoBuildStamp != null) {
      defines.add(CppConfiguration.FDO_STAMP_MACRO + "=\"" + fdoBuildStamp + "\"");
    }

    return defines
        .build()
        .stream()
        .map(
            define ->
                define
                    .replaceAll(labelPattern, labelReplacement)
                    .replaceAll(outputPathPattern, outputReplacement))
        .collect(ImmutableList.toImmutableList());
  }

  private static Variables getVariables(
      Artifact sourceFile,
      Artifact outputFile,
      String labelReplacement,
      String outputReplacement,
      Iterable<String> additionalLinkstampDefines,
      ImmutableList<Artifact> buildInfoHeaderArtifacts,
      FeatureConfiguration featureConfiguration,
      CppConfiguration cppConfiguration,
      CcToolchainProvider ccToolchainProvider,
      boolean needsPic,
      String fdoBuildStamp,
      boolean codeCoverageEnabled) {
    // TODO(b/34761650): Remove all this hardcoding by separating a full blown compile action.
    Preconditions.checkArgument(
        featureConfiguration.actionIsConfigured(CppCompileAction.LINKSTAMP_COMPILE));

    Variables.Builder variables = new Variables.Builder(ccToolchainProvider.getBuildVariables());
    // We need to force inclusion of build_info headers
    variables.addStringSequenceVariable(
        CppModel.INCLUDES_VARIABLE_NAME,
        buildInfoHeaderArtifacts
            .stream()
            .map(Artifact::getExecPathString)
            .collect(ImmutableList.toImmutableList()));
    // Input/Output files.
    variables.addStringVariable(CppModel.SOURCE_FILE_VARIABLE_NAME, sourceFile.getExecPathString());
    variables.addStringVariable(CppModel.OUTPUT_FILE_VARIABLE_NAME, outputFile.getExecPathString());
    variables.addStringVariable(
        CppModel.OUTPUT_OBJECT_FILE_VARIABLE_NAME, outputFile.getExecPathString());
    // Include directories for (normal includes with ".", empty quote- and system- includes).
    variables.addStringSequenceVariable(
        CppModel.INCLUDE_PATHS_VARIABLE_NAME, ImmutableList.of("."));
    variables.addStringSequenceVariable(
        CppModel.QUOTE_INCLUDE_PATHS_VARIABLE_NAME, ImmutableList.of());
    variables.addStringSequenceVariable(
        CppModel.SYSTEM_INCLUDE_PATHS_VARIABLE_NAME, ImmutableList.of());
    // Legacy flags coming from fields such as compiler_flag
    variables.addLazyStringSequenceVariable(
        CppModel.LEGACY_COMPILE_FLAGS_VARIABLE_NAME,
        CppModel.getLegacyCompileFlagsSupplier(
            cppConfiguration, sourceFile.getExecPathString(), ImmutableSet.of()));
    // Unfilterable flags coming from unfiltered_cxx_flag fields
    variables.addLazyStringSequenceVariable(
        CppModel.UNFILTERED_COMPILE_FLAGS_VARIABLE_NAME,
        CppModel.getUnfilteredCompileFlagsSupplier(ccToolchainProvider, ImmutableSet.of()));
    // Collect all preprocessor defines, and in each replace ${LABEL} by labelReplacements, and
    // ${OUTPUT_PATH} with outputPathReplacement.
    variables.addStringSequenceVariable(
        CppModel.PREPROCESSOR_DEFINES_VARIABLE_NAME,
        computeAllLinkstampDefines(
            labelReplacement,
            outputReplacement,
            additionalLinkstampDefines,
            cppConfiguration,
            fdoBuildStamp,
            codeCoverageEnabled));
    // For dynamic libraries, produce position independent code.
    if (needsPic) {
      variables.addStringVariable(CppModel.PIC_VARIABLE_NAME, "");
    }

    return variables.build();
  }
}
