// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaToolchainStarlarkApiProviderApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;

final class FakeJavaToolchainStarlarkApiProviderApi implements JavaToolchainStarlarkApiProviderApi {

  @Override
  public String getSourceVersion() {
    return null;
  }

  @Override
  public String getTargetVersion() {
    return null;
  }

  @Override
  public FileApi getSingleJar() {
    return null;
  }

  @Override
  public Depset getStarlarkBootclasspath() {
    return null;
  }

  @Override
  public Sequence<String> getStarlarkJvmOptions() {
    return null;
  }

  @Override
  public FilesToRunProviderApi<?> getJacocoRunner() {
    return null;
  }

  @Override
  public Depset getStarlarkTools() {
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

  private FakeJavaToolchainStarlarkApiProviderApi() {}
}
