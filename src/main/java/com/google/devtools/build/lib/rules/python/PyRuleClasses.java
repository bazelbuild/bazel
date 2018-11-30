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

import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;

/**
 * Rule definitions for Python rules.
 */
public class PyRuleClasses {

  public static final FileType PYTHON_SOURCE = FileType.of(".py", ".py3");

  /**
   * Input for {@link RuleClass.Builder#cfg(RuleTransitionFactory)}: if {@link
   * PythonOptions#forcePython} is unset, sets the Python version according to the rule's default
   * Python version. Assumes the rule has the expected attribute for this setting.
   *
   * <p>Since this is a configuration transition, this propagates to the rules' transitive deps.
   */
  public static final RuleTransitionFactory DEFAULT_PYTHON_VERSION_TRANSITION =
      (rule) ->
          new PythonVersionTransition(
              // In case of a parse error, this will return null, which means that the transition
              // would use the hard-coded default (PythonVersion#getDefaultTargetValue). But the
              // attribute is already validated to allow only PythonVersion#getTargetStrings anyway.
              PythonVersion.parse(
                  RawAttributeMapper.of(rule).get("default_python_version", Type.STRING),
                  PythonVersion.getAllValues()));
}
