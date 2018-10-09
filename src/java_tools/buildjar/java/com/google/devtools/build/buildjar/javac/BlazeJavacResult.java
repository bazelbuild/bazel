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
import com.google.devtools.build.buildjar.javac.statistics.BlazeJavacStatistics;
import javax.annotation.Nullable;

/** The result of a single compilation performed by {@link BlazeJavacMain}. */
public class BlazeJavacResult {

  private final boolean ok;
  private final ImmutableList<FormattedDiagnostic> diagnostics;
  private final String output;
  private final BlazeJavaCompiler compiler;
  private final BlazeJavacStatistics statistics;

  public static BlazeJavacResult ok() {
    return createFullResult(true, ImmutableList.of(), "", null, BlazeJavacStatistics.empty());
  }

  public static BlazeJavacResult error(String message) {
    return createFullResult(false, ImmutableList.of(), message, null, BlazeJavacStatistics.empty());
  }

  private BlazeJavacResult(
      boolean ok,
      ImmutableList<FormattedDiagnostic> diagnostics,
      String output,
      @Nullable BlazeJavaCompiler compiler,
      BlazeJavacStatistics statistics) {
    this.ok = ok;
    this.diagnostics = diagnostics;
    this.output = output;
    this.compiler = compiler;
    this.statistics = statistics;
  }

  public static BlazeJavacResult createFullResult(
      boolean ok,
      ImmutableList<FormattedDiagnostic> diagnostics,
      String output,
      BlazeJavaCompiler compiler,
      BlazeJavacStatistics statistics) {
    return new BlazeJavacResult(ok, diagnostics, output, compiler, statistics);
  }

  public boolean isOk() {
    return ok;
  }

  public ImmutableList<FormattedDiagnostic> diagnostics() {
    return diagnostics;
  }

  public String output() {
    return output;
  }

  public BlazeJavacStatistics statistics() {
    return statistics;
  }

  @VisibleForTesting
  public BlazeJavaCompiler compiler() {
    return compiler;
  }
}
