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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.java.turbine.javac.ZipOutputFileManager.OutputFileObject;

import com.sun.tools.javac.util.Context;

import java.io.StringWriter;

/** The output from a {@link JavacTurbineCompiler} compilation. */
class JavacTurbineCompileResult {

  private final ImmutableMap<String, OutputFileObject> files;
  private final boolean success;
  private final StringWriter sb;
  private final Context context;

  JavacTurbineCompileResult(
      ImmutableMap<String, OutputFileObject> files,
      boolean success,
      StringWriter sb,
      Context context) {
    this.files = files;
    this.success = success;
    this.sb = sb;
    this.context = context;
  }

  /** True iff the compilation succeeded. */
  boolean success() {
    return success;
  }

  /** The stderr from the compilation. */
  String output() {
    return sb.toString();
  }

  /** The files produced by the compilation's {@link ZipOutputFileManager}. */
  ImmutableMap<String, OutputFileObject> files() {
    return files;
  }

  /** The compilation context, may by inspected by integration tests. */
  @VisibleForTesting
  Context context() {
    return context;
  }
}
