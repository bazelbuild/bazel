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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate.OutputPathMapper;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidIdeInfoProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidSdkProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidStarlarkCommonApi;
import java.io.Serializable;
import java.util.List;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkInt;

/** Common utilities for Starlark rules related to Android. */
public class AndroidStarlarkCommon
    implements AndroidStarlarkCommonApi<
        Artifact, JavaInfo, FilesToRunProvider, ConstraintValueInfo, StarlarkRuleContext> {

  @Override
  public String getSourceDirectoryRelativePathFromResource(Artifact resource) {
    return AndroidCommon.getSourceDirectoryRelativePathFromResource(resource).toString();
  }

  /**
   * TODO(b/14473160): Provides a Starlark compatibility layer for the sourceless deps bug. When a
   * sourceless target is defined, the deps of the target are implicitly exported. Specifically only
   * the {@link JavaCompilationArgsProvider} is propagated. This method takes the existing JavaInfo
   * and produces a new one, only containing the {@link JavaCompilationArgsProvider} to be added to
   * the exports field of the java_common.compile method. Remove this method once the bug has been
   * fixed.
   */
  @Override
  public JavaInfo enableImplicitSourcelessDepsExportsCompatibility(Info javaInfo, boolean neverlink)
      throws RuleErrorException {
    JavaCompilationArgsProvider.ClasspathType type =
        neverlink
            ? JavaCompilationArgsProvider.ClasspathType.COMPILE_ONLY
            : JavaCompilationArgsProvider.ClasspathType.BOTH;
    JavaInfo.Builder builder = JavaInfo.Builder.create();
    JavaInfo.PROVIDER
        .wrap(javaInfo)
        .compilationArgsProvider()
        .ifPresent(
            args ->
                builder.javaCompilationArgs(
                    JavaCompilationArgsProvider.builder().addExports(args, type).build()));
    return builder.setNeverlink(neverlink).build();
  }

  @Override
  public void createDexMergerActions(
      StarlarkRuleContext starlarkRuleContext,
      Artifact output,
      Artifact input,
      Sequence<?> dexopts, // <String> expected.
      FilesToRunProvider dexmerger,
      StarlarkInt minSdkVersion,
      Object desugarGlobals)
      throws EvalException, RuleErrorException {
    createTemplatedMergerActions(
        starlarkRuleContext.getRuleContext(),
        (SpecialArtifact) output,
        (SpecialArtifact) input,
        Sequence.cast(dexopts, String.class, "dexopts"),
        dexmerger,
        minSdkVersion.toInt("min_sdk_version"),
        desugarGlobals);
  }

  /**
   * Sets up a monodex {@code $dexmerger} actions for each dex archive in the given tree artifact
   * and puts the outputs in a tree artifact.
   */
  private static void createTemplatedMergerActions(
      RuleContext ruleContext,
      SpecialArtifact outputTree,
      SpecialArtifact inputTree,
      List<String> dexopts,
      FilesToRunProvider executable,
      int minSdkVersion,
      Object desugarGlobals) {
    SpawnActionTemplate.Builder dexmerger =
        new SpawnActionTemplate.Builder(inputTree, outputTree)
            .setExecutable(executable)
            .setMnemonics("DexShardsToMerge", "DexMerger")
            .setOutputPathMapper(
                (OutputPathMapper & Serializable) TreeFileArtifact::getParentRelativePath);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addPlaceholderTreeArtifactExecPath("--input", inputTree)
            .addPlaceholderTreeArtifactExecPath("--output", outputTree)
            .add("--multidex=given_shard")
            .addAll(
                AndroidCommon.mergerDexopts(
                    ruleContext,
                    Iterables.filter(
                        dexopts, Predicates.not(Predicates.equalTo("--minimal-main-dex")))));
    if (minSdkVersion > 0) {
      commandLine.add("--min_sdk_version", Integer.toString(minSdkVersion));
    }
    Artifact desugarGlobalsArtifact =
        AndroidStarlarkData.fromNoneable(desugarGlobals, Artifact.class);
    if (desugarGlobalsArtifact != null) {
      dexmerger.addCommonInputs(ImmutableList.of(desugarGlobalsArtifact));
      commandLine.addPath("--global_synthetics_path", desugarGlobalsArtifact.getExecPath());
    }
    dexmerger.setCommandLineTemplate(commandLine.build());
    ruleContext.registerAction(dexmerger.build(ruleContext.getActionOwner()));
  }

  @StarlarkMethod(name = AndroidIdeInfoProviderApi.NAME, structField = true, documented = false)
  public AndroidIdeInfoProvider.Provider getAndroidIdeInfoProvider() {
    return AndroidIdeInfoProvider.PROVIDER;
  }

  @StarlarkMethod(name = AndroidSdkProviderApi.NAME, structField = true, documented = false)
  public AndroidSdkProvider.Provider getAndroidSdkInfoProvider() {
    return AndroidSdkProvider.PROVIDER;
  }
}
