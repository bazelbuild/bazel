// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.python;

/** Enumerates the different modes of Python precompilation. */
public enum PrecompilePythonMode {
  /** No Python precompilation should take place. */
  NONE,

  /** Only the _pb/_pb2.py files generated from protocol buffers should be precompiled. */
  PROTO,

  /** Compiles all Python files, but removes the .py sources from the runfiles. */
  ONLY,

  /** Compiles all Python files, and leaves the .py sources in the runfiles. */
  ALL,

  /** The default mode for the platform. */
  DEFAULT;

  public boolean shouldPrecompileProtos() {
    return this != NONE;
  };

  public boolean shouldPrecompilePythonSources() {
    return this == ONLY || this == ALL;
  };
}
