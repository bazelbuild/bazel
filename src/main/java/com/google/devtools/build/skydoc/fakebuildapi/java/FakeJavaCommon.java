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
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaCommonApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.skydoc.fakebuildapi.FakeProviderApi;
import javax.annotation.Nullable;

/** Fake implementation of {@link JavaCommonApi}. */
public class FakeJavaCommon
    implements JavaCommonApi<
        FileApi, FakeJavaInfo, SkylarkRuleContextApi, SkylarkActionFactoryApi> {

  @Override
  public FakeJavaInfo create(@Nullable Object actionsUnchecked, Object compileTimeJars,
      Object runtimeJars, Boolean useIjar, @Nullable Object javaToolchainUnchecked,
      Object transitiveCompileTimeJars, Object transitiveRuntimeJars, Object sourceJars,
      Location location, Environment environment) throws EvalException {
    return new FakeJavaInfo();
  }

  @Override
  public ProviderApi getJavaProvider() {
    return new FakeProviderApi();
  }

  @Override
  public FakeJavaInfo createJavaCompileAction(
      SkylarkRuleContextApi skylarkRuleContext,
      SkylarkList<FileApi> sourceJars,
      SkylarkList<FileApi> sourceFiles,
      FileApi outputJar,
      SkylarkList<String> javacOpts,
      SkylarkList<FakeJavaInfo> deps,
      SkylarkList<FakeJavaInfo> exports,
      SkylarkList<FakeJavaInfo> plugins,
      SkylarkList<FakeJavaInfo> exportedPlugins,
      String strictDepsMode,
      Object javaToolchain,
      Object hostJavabase,
      SkylarkList<FileApi> sourcepathEntries,
      SkylarkList<FileApi> resources,
      Boolean neverlink,
      Location loc,
      Environment environment)
      throws EvalException, InterruptedException {
    return new FakeJavaInfo();
  }

  @Override
  public FileApi runIjar(
      SkylarkActionFactoryApi actions,
      FileApi jar,
      Object targetLabel,
      Object javaToolchain,
      Location location,
      SkylarkSemantics semantics)
      throws EvalException {
    return null;
  }

  @Override
  public FileApi stampJar(
      SkylarkActionFactoryApi actions,
      FileApi jar,
      Label targetLabel,
      Object javaToolchain,
      Location location,
      SkylarkSemantics semantics)
      throws EvalException {
    return null;
  }

  @Override
  public FileApi packSources(
      SkylarkActionFactoryApi actions,
      FileApi outputJar,
      SkylarkList<FileApi> sourceFiles,
      SkylarkList<FileApi> sourceJars,
      Object javaToolchain,
      Object hostJavabase,
      Location location,
      SkylarkSemantics semantics)
      throws EvalException {
    return null;
  }

  @Override
  public ImmutableList<String> getDefaultJavacOpts(
      SkylarkRuleContextApi skylarkRuleContext,
      String javaToolchainAttr,
      Location loc,
      SkylarkSemantics semantics)
      throws EvalException {
    return ImmutableList.of();
  }

  @Override
  public FakeJavaInfo mergeJavaProviders(SkylarkList<FakeJavaInfo> providers) {
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
}
