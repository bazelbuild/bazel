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
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
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
   * A value set of the target and sentinel values that doesn't mention the sentinel in error
   * messages.
   */
  public static final AllowedValueSet TARGET_PYTHON_ATTR_VALUE_SET =
      new AllowedValueSet(PythonVersion.TARGET_AND_SENTINEL_STRINGS) {
        @Override
        public String getErrorReason(Object value) {
          return String.format("has to be one of 'PY2' or 'PY3' instead of '%s'", value);
        }
      };

  /**
   * Returns a rule transition factory for Python binary rules and other rules that may change the
   * Python version.
   *
   * <p>The factory makes a transition to set the Python version to the value specified by the
   * rule's {@code python_version} attribute if it is given, or otherwise the {@code
   * default_python_version} attribute if it is given, or otherwise the default value passed into
   * this function.
   *
   * <p>The factory throws {@link IllegalArgumentException} if used on a rule whose {@link
   * RuleClass} does not define both attributes. If both are defined, but one of their values cannot
   * be parsed as a Python version, the given default value is used as a fallback instead; in this
   * case it is up to the rule's analysis phase ({@link PyCommon#validateTargetPythonVersionAttr})
   * to report an attribute error to the user. This case should be prevented by attribute validation
   * if the rule is defined correctly.
   */
  public static RuleTransitionFactory makeVersionTransition(PythonVersion defaultVersion) {
    return (rule) -> {
      AttributeMap attrs = RawAttributeMapper.of(rule);
      // Fail fast if we're used on an ill-defined rule class.
      Preconditions.checkArgument(
          attrs.has(PyCommon.PYTHON_VERSION_ATTRIBUTE, Type.STRING)
              && attrs.has(PyCommon.DEFAULT_PYTHON_VERSION_ATTRIBUTE, Type.STRING),
          "python version transitions require that the RuleClass define both "
              + "'default_python_version' and 'python_version'");
      // Attribute validation should enforce that the attribute string value is either a target
      // value ("PY2" or "PY3") or the sentinel value ("_INTERNAL_SENTINEL"). But just in case,
      // we'll, treat an invalid value as the default value rather than propagate an unchecked
      // exception in this context. That way the user can at least get a clean error message
      // instead of a crash.
      PythonVersion version;
      try {
        version = PyCommon.readPythonVersionFromAttributes(attrs, defaultVersion);
      } catch (IllegalArgumentException ex) {
        version = defaultVersion;
      }
      return new PythonVersionTransition(version);
    };
  }

  /**
   * A Python version transition that sets the version as specified by the target's attributes, with
   * a default of {@link PythonVersion#DEFAULT_TARGET_VALUE}.
   */
  public static final RuleTransitionFactory VERSION_TRANSITION =
      makeVersionTransition(PythonVersion.DEFAULT_TARGET_VALUE);
}
