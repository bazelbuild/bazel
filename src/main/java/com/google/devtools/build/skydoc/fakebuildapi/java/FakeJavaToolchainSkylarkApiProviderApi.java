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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaToolchainSkylarkApiProviderApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

final class FakeJavaToolchainSkylarkApiProviderApi implements JavaToolchainSkylarkApiProviderApi {

  @Override
  public String getSourceVersion() {
    return null;
  }

  @Override
  public String getTargetVersion() {
    return null;
  }

  @Override
  public FileApi getJavacJar() {
    return null;
  }

  @Override
  public FileApi getSingleJar() {
    return null;
  }

  @Override
  public SkylarkNestedSet getSkylarkBootclasspath() {
    return null;
  }

  @Override
  public SkylarkList<String> getSkylarkJvmOptions() {
    return null;
  }

  @Override
  public SkylarkNestedSet getSkylarkTools() {
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

  private FakeJavaToolchainSkylarkApiProviderApi() {}
}
