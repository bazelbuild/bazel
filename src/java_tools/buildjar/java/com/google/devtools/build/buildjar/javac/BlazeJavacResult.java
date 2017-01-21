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

package com.google.devtools.build.buildjar.javac;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.sun.tools.javac.main.Main.Result;

/** The result of a single compilation performed by {@link BlazeJavacMain}. */
public class BlazeJavacResult {

  private final Result javacResult;
  private final ImmutableList<FormattedDiagnostic> diagnostics;
  private final String output;
  private final BlazeJavaCompiler compiler;

  public static BlazeJavacResult ok() {
    return new BlazeJavacResult(Result.OK, ImmutableList.of(), "", null);
  }

  public static BlazeJavacResult error(String message) {
    return new BlazeJavacResult(Result.ERROR, ImmutableList.of(), message, null);
  }

  public BlazeJavacResult(
      Result javacResult,
      ImmutableList<FormattedDiagnostic> diagnostics,
      String output,
      BlazeJavaCompiler compiler) {
    this.javacResult = javacResult;
    this.diagnostics = diagnostics;
    this.output = output;
    this.compiler = compiler;
  }

  public Result javacResult() {
    return javacResult;
  }

  public ImmutableList<FormattedDiagnostic> diagnostics() {
    return diagnostics;
  }

  public String output() {
    return output;
  }

  @VisibleForTesting
  public BlazeJavaCompiler compiler() {
    return compiler;
  }
}
