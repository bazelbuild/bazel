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

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaAnnotationProcessingApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaCompilationInfoProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaRuleOutputJarsProviderApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkThread;

/**
 * Fake implementation of {@link JavaInfoApi}.
 */
public class FakeJavaInfo implements JavaInfoApi<FileApi> {

  @Override
  public Depset /*<File>*/ getTransitiveRuntimeJars() {
    return null;
  }

  @Override
  public Depset getTransitiveCompileTimeJars() {
    return null;
  }

  @Override
  public Depset getCompileTimeJars() {
    return null;
  }

  @Override
  public Depset getFullCompileTimeJars() {
    return null;
  }

  @Override
  public Sequence<FileApi> getSourceJars() {
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
  public Sequence<FileApi> getRuntimeOutputJars() {
    return null;
  }

  @Override
  public Depset /*<File>*/ getTransitiveDeps() {
    return null;
  }

  @Override
  public Depset /*<File>*/ getTransitiveRuntimeDeps() {
    return null;
  }

  @Override
  public Depset /*<File>*/ getTransitiveSourceJars() {
    return null;
  }

  @Override
  public Depset /*<Label>*/ getTransitiveExports() {
    return null;
  }

  @Override
  public String toProto() throws EvalException {
    return "";
  }

  @Override
  public String toJson() throws EvalException {
    return "";
  }

  @Override
  public void repr(Printer printer) {}

  /**
   * Fake implementation of {@link JavaInfoProviderApi}.
   */
  public static class FakeJavaInfoProvider implements JavaInfoProviderApi {

    @Override
    public JavaInfoApi<?> javaInfo(
        FileApi outputJarApi,
        Object compileJarApi,
        Object sourceJarApi,
        Boolean neverlink,
        Sequence<?> deps,
        Sequence<?> runtimeDeps,
        Sequence<?> exports,
        Object jdepsApi,
        StarlarkThread thread)
        throws EvalException {
      return new FakeJavaInfo();
    }

    @Override
    public void repr(Printer printer) {}
  }
}
