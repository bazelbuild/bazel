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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaAnnotationProcessingApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaCompilationInfoProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaRuleOutputJarsProviderApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * Fake implementation of {@link JavaInfoApi}.
 */
public class FakeJavaInfo implements JavaInfoApi<FileApi> {

  @Override
  public SkylarkNestedSet getTransitiveRuntimeJars() {
    return null;
  }

  @Override
  public SkylarkNestedSet getTransitiveCompileTimeJars() {
    return null;
  }

  @Override
  public SkylarkNestedSet getCompileTimeJars() {
    return null;
  }

  @Override
  public SkylarkNestedSet getFullCompileTimeJars() {
    return null;
  }

  @Override
  public SkylarkList<FileApi> getSourceJars() {
    return null;
  }

  @Override
  public JavaRuleOutputJarsProviderApi<?> getOutputJars() {
    return null;
  }

  @Override
  public JavaAnnotationProcessingApi<?> getGenJarsProvider() {
    return null;
  }

  @Override
  public JavaCompilationInfoProviderApi<?> getCompilationInfoProvider() {
    return null;
  }

  @Override
  public SkylarkList<FileApi> getRuntimeOutputJars() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveDeps() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveRuntimeDeps() {
    return null;
  }

  @Override
  public NestedSet<FileApi> getTransitiveSourceJars() {
    return null;
  }

  @Override
  public NestedSet<Label> getTransitiveExports() {
    return null;
  }

  @Override
  public String toProto(Location loc) throws EvalException {
    return "";
  }

  @Override
  public String toJson(Location loc) throws EvalException {
    return "";
  }

  @Override
  public void repr(SkylarkPrinter printer) {}

  /**
   * Fake implementation of {@link JavaInfoProviderApi}.
   */
  public static class FakeJavaInfoProvider implements JavaInfoProviderApi {

    @Override
    public JavaInfoApi<?> javaInfo(FileApi outputJarApi, Object compileJarApi, Object sourceJarApi,
        Boolean neverlink, SkylarkList<?> deps, SkylarkList<?> runtimeDeps, SkylarkList<?> exports,
        Object actionsApi, Object sourcesApi, Object sourceJarsApi, Object useIjarApi,
        Object javaToolchainApi, Object hostJavabaseApi, Object jdepsApi, Location loc,
        Environment env) throws EvalException {
      return new FakeJavaInfo();
    }

    @Override
    public void repr(SkylarkPrinter printer) {}
  }
}
