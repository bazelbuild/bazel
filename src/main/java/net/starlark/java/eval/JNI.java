// Copyright 2020 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

final class JNI {
  private JNI() {} // uninstantiable

  static void load() {
    try {
      System.loadLibrary("cpu_profiler");
    } catch (UnsatisfiedLinkError ex) {
      // Ignore, deferring the error until a C function is called, if ever.
      // Without this hack //src/test/shell/bazel:bazel_bootstrap_distfile_test
      // fails with an utterly uninformative error.
      // TODO(adonovan): remove try/catch once that test is fixed.
    }
  }
}
