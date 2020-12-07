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

package com.google.devtools.build.lib.packages.util;

import com.google.devtools.build.lib.rules.python.PythonSemantics;
import java.io.IOException;

/**
 * Creates mock BUILD files required for the python rules.
 */
public abstract class MockPythonSupport {

  /** Setup the support for building Python. */
  public abstract void setup(MockToolsConfig config) throws IOException;

  /**
   * Setup support for, and return the string label of, a target that can be passed to {@code
   * --python_top} that causes the {@code py_runtime} consumed by Python rules to be the given
   * {@code pyRuntimeLabel}.
   */
  public abstract String createPythonTopEntryPoint(MockToolsConfig config, String pyRuntimeLabel)
      throws IOException;

  public abstract PythonSemantics getPythonSemantics();
}
