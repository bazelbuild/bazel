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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;

/** Rule definitions for Python rules. */
public class PyRuleClasses {

  public static final FileType PYTHON_SOURCE = FileType.of(".py", ".py3");

  /**
   * A rule transition factory for Python binary rules and other rules that may change the Python
   * version.
   *
   * <p>This sets the Python version to the value specified by {@code python_version} if it is given
   * explicitly, or the (possibly default) value of {@code default_python_version} otherwise.
   *
   * <p>The transition throws {@link IllegalArgumentException} if used on a rule whose {@link
   * RuleClass} does not define both attributes. If both are defined, but the one whose value is to
   * be read cannot be parsed as a Python version, {@link PythonVersion#DEFAULT_TARGET_VALUE} is
   * used instead; in this case it is up to the rule's analysis phase (in {@link PyCommon}) to
   * report an attribute error to the user.
   */
  public static final RuleTransitionFactory PYTHON_VERSION_TRANSITION =
      (rule) -> {
        AttributeMap attrs = RawAttributeMapper.of(rule);
        Preconditions.checkArgument(
            attrs.has(PyCommon.PYTHON_VERSION_ATTRIBUTE, Type.STRING)
                && attrs.has(PyCommon.DEFAULT_PYTHON_VERSION_ATTRIBUTE, Type.STRING),
            "python version transitions require that the RuleClass define both "
                + "'default_python_version' and 'python_version'");

        String versionString =
            attrs.isAttributeValueExplicitlySpecified(PyCommon.PYTHON_VERSION_ATTRIBUTE)
                ? attrs.get(PyCommon.PYTHON_VERSION_ATTRIBUTE, Type.STRING)
                : attrs.get(PyCommon.DEFAULT_PYTHON_VERSION_ATTRIBUTE, Type.STRING);

        // It should be a target value ("PY2" or "PY3"), and if not that should be caught by
        // attribute validation. But just in case, we'll treat an invalid value as null (which means
        // "use the hard-coded default version") rather than propagate an unchecked exception in
        // this context. That way the user can at least get a clean error message instead of a
        // crash.
        PythonVersion version;
        try {
          version = PythonVersion.parseTargetValue(versionString);
        } catch (IllegalArgumentException ex) {
          version = null;
        }
        return new PythonVersionTransition(version);
      };
}
