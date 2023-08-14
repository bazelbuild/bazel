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

import com.google.devtools.build.lib.analysis.Runfiles;

/**
 * Pluggable semantics for Python rules.
 *
 * <p>A new instance of this class is created for each configured target, therefore, it is allowed
 * to keep state.
 */
public interface PythonSemantics {
  /**
   * Called when building executables or packages to fill in missing empty __init__.py files if the
   * --incompatible_default_to_explicit_init_py has not yet been enabled. This usually returns a
   * public static final reference, code is free to use that directly on specific implementations
   * instead of making this call.
   */
  Runfiles.EmptyFilesSupplier getEmptyRunfilesSupplier();
}
