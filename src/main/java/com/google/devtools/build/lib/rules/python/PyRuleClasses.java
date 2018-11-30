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

/** Rule definitions for Python rules. */
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
      (rule) -> {
        String attrDefault = RawAttributeMapper.of(rule).get("default_python_version", Type.STRING);
        // It should be a target value ("PY2" or "PY3"), and if not that should be caught by
        // attribute validation. But just in case, we'll treat an invalid value as null (which means
        // "use the hard-coded default version") rather than propagate an unchecked exception in
        // this context.
        PythonVersion version = null;
        // Should be non-null because this transition shouldn't be used on rules without the attr.
        if (attrDefault != null) {
          try {
            version = PythonVersion.parseTargetValue(attrDefault);
          } catch (IllegalArgumentException ex) {
            // Parsing error.
          }
        }
        return new PythonVersionTransition(version);
      };
}
