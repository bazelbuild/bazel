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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidSplitTransitionApi;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidStarlarkCommonApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** Common utilities for Starlark rules related to Android. */
public class AndroidStarlarkCommon
    implements AndroidStarlarkCommonApi<
        Artifact, JavaInfo, FilesToRunProvider, ConstraintValueInfo, StarlarkRuleContext> {

  @Override
  public AndroidDeviceBrokerInfo createDeviceBrokerInfo(String deviceBrokerType) {
    return new AndroidDeviceBrokerInfo(deviceBrokerType);
  }

  @Override
  public String getSourceDirectoryRelativePathFromResource(Artifact resource) {
    return AndroidCommon.getSourceDirectoryRelativePathFromResource(resource).toString();
  }

  @Override
  public AndroidSplitTransitionApi getAndroidSplitTransition() {
    return AndroidRuleClasses.ANDROID_SPLIT_TRANSITION;
  }

  @Override
  public StarlarkExposedRuleTransitionFactory getAndroidPlatformsTransition() {
    return new AndroidPlatformsTransition.AndroidPlatformsTransitionFactory();
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
      FilesToRunProvider dexmerger)
      throws EvalException, RuleErrorException {
    AndroidBinary.createTemplatedMergerActions(
        starlarkRuleContext.getRuleContext(),
        (SpecialArtifact) output,
        (SpecialArtifact) input,
        Sequence.cast(dexopts, String.class, "dexopts"),
        dexmerger);
  }
}
