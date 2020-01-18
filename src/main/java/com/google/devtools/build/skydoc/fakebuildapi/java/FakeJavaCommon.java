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

package com.google.devtools.build.skydoc.fakebuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaCommonApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaToolchainSkylarkApiProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.skydoc.fakebuildapi.FakeProviderApi;

/** Fake implementation of {@link JavaCommonApi}. */
public class FakeJavaCommon
    implements JavaCommonApi<
        FileApi,
        FakeJavaInfo,
        FakeJavaToolchainSkylarkApiProviderApi,
        FakeJavaRuntimeInfoApi,
        ConstraintValueInfoApi,
        SkylarkRuleContextApi<ConstraintValueInfoApi>,
        SkylarkActionFactoryApi> {

  @Override
  public ProviderApi getJavaProvider() {
    return new FakeProviderApi();
  }

  @Override
  public FakeJavaInfo createJavaCompileAction(
      SkylarkRuleContextApi<ConstraintValueInfoApi> skylarkRuleContext,
      Sequence<?> sourceJars,
      Sequence<?> sourceFiles,
      FileApi outputJar,
      Object outputSourceJar,
      Sequence<?> javacOpts,
      Sequence<?> deps,
      Sequence<?> experimentalLocalCompileTimeDeps,
      Sequence<?> exports,
      Sequence<?> plugins,
      Sequence<?> exportedPlugins,
      Sequence<?> annotationProcessorAdditionalInputs,
      Sequence<?> annotationProcessorAdditionalOutputs,
      String strictDepsMode,
      FakeJavaToolchainSkylarkApiProviderApi javaToolchain,
      FakeJavaRuntimeInfoApi hostJavabase,
      Sequence<?> sourcepathEntries,
      Sequence<?> resources,
      Boolean neverlink,
      Location loc,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    return new FakeJavaInfo();
  }

  @Override
  public FileApi runIjar(
      SkylarkActionFactoryApi actions,
      FileApi jar,
      Object targetLabel,
      FakeJavaToolchainSkylarkApiProviderApi javaToolchain,
      Location location)
      throws EvalException {
    return null;
  }

  @Override
  public FileApi stampJar(
      SkylarkActionFactoryApi actions,
      FileApi jar,
      Label targetLabel,
      FakeJavaToolchainSkylarkApiProviderApi javaToolchain,
      Location location)
      throws EvalException {
    return null;
  }

  @Override
  public FileApi packSources(
      SkylarkActionFactoryApi actions,
      FileApi outputJar,
      Sequence<?> sourceFiles,
      Sequence<?> sourceJars,
      FakeJavaToolchainSkylarkApiProviderApi javaToolchain,
      FakeJavaRuntimeInfoApi hostJavabase,
      Location location)
      throws EvalException {
    return null;
  }

  @Override
  public ImmutableList<String> getDefaultJavacOpts(
      FakeJavaToolchainSkylarkApiProviderApi javaToolchain, Location loc) throws EvalException {
    return ImmutableList.of();
  }

  @Override
  public FakeJavaInfo mergeJavaProviders(Sequence<?> providers) {
    return new FakeJavaInfo();
  }

  @Override
  public FakeJavaInfo makeNonStrict(FakeJavaInfo javaInfo) {
    return new FakeJavaInfo();
  }

  @Override
  public ProviderApi getJavaToolchainProvider() {
    return new FakeProviderApi();
  }

  @Override
  public ProviderApi getJavaRuntimeProvider() {
    return new FakeProviderApi();
  }

  @Override
  public boolean isJavaToolchainResolutionEnabled(
      SkylarkRuleContextApi<ConstraintValueInfoApi> ruleContext) {
    return false;
  }

  @Override
  public ProviderApi getMessageBundleInfo() {
    return new FakeProviderApi();
  }

  @Override
  public FakeJavaInfo addConstraints(FakeJavaInfo javaInfo, Sequence<?> constraints) {
    return new FakeJavaInfo();
  }

  @Override
  public FakeJavaInfo removeAnnotationProcessors(FakeJavaInfo javaInfo) {
    return new FakeJavaInfo();
  }

  @Override
  public Depset /*<FileApi>*/ getCompileTimeJavaDependencyArtifacts(FakeJavaInfo javaInfo) {
    return null;
  }

  @Override
  public FakeJavaInfo addCompileTimeJavaDependencyArtifacts(
      FakeJavaInfo javaInfo, Sequence<?> compileTimeJavaDependencyArtifacts) {
    return new FakeJavaInfo();
  }

  @Override
  public Label getJavaToolchainLabel(
      JavaToolchainSkylarkApiProviderApi toolchain, Location location) throws EvalException {
    return null;
  }
}
