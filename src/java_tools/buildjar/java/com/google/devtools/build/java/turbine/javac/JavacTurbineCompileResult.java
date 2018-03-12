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

package com.google.devtools.build.java.turbine.javac;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.sun.tools.javac.util.Context;

/** The output from a {@link JavacTurbineCompiler} compilation. */
class JavacTurbineCompileResult {


  enum Status {
    OK, ERROR
  }

  private final ImmutableMap<String, byte[]> files;
  private final Status status;
  private final String output;
  private final ImmutableList<FormattedDiagnostic> diagnostics;
  private final Context context;

  JavacTurbineCompileResult(
      ImmutableMap<String, byte[]> files,
      Status status,
      String output,
      ImmutableList<FormattedDiagnostic> diagnostics,
      Context context) {
    this.files = files;
    this.status = status;
    this.output = output;
    this.diagnostics = diagnostics;
    this.context = context;
  }

  /** True iff the compilation succeeded. */
  boolean success() {
    return status == Status.OK;
  }

  /** The stderr from the compilation. */
  String output() {
    return output;
  }

  /** The diagnostics from the compilation. */
  ImmutableList<FormattedDiagnostic> diagnostics() {
    return diagnostics;
  }

  /** The files produced by the compilation. */
  ImmutableMap<String, byte[]> files() {
    return files;
  }

  /** The compilation context, may by inspected by integration tests. */
  @VisibleForTesting
  Context context() {
    return context;
  }
}
