// Copyright 2015 The Bazel Authors. All rights reserved.
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

/**
 * Various implicit attributes names that Python rules use, such as the Python proto runtime.
 */
public class PythonImplicitAttributes {
  public static final String DEFAULT_PYTHON_PRECOMPILE_ATTRIBUTE = "$python_precompile";
  public static final String DEFAULT_HOST_PYTHON2_RUNTIME_ATTRIBUTE = ":host_python2_runtime";
}
