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

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import java.util.List;

/**
 * An enum representing Python major versions.
 *
 * <p>This enum has two interpretations. The "target" interpretation is when this enum is used in a
 * command line flag or in a rule attribute to denote a particular version of the Python language.
 * Only {@code PY2} and {@code PY3} can be used as target values. The "sources" interpretation is
 * when this enum is used to denote the degree of compatibility of source code with the target
 * values.
 */
public enum PythonVersion {

  // TODO(#6445): Remove PY2ONLY and PY3ONLY.

  /**
   * Target value Python 2. Represents source code that is naturally compatible with Python 2.
   *
   * <p><i>Deprecated meaning:</i> Also indicates that source code is compatible with Python 3 under
   * 2to3 transformation. 2to3 transformation is not implemented in Bazel and this meaning will be
   * removed from Bazel (#1393).
   */
  PY2,

  /**
   * Target value Python 3. Represents source code that is naturally compatible with Python 3.
   *
   * <p><i>Deprecated meaning:</i> Also indicates that source code is compatible with Python 2 under
   * 3to2 transformation. 3to2 transformation was never implemented and this meaning should not be
   * relied on.
   */
  PY3,

  /**
   * Represents source code that is naturally compatible with both Python 2 and Python 3, i.e. code
   * that lies in the intersection of both languages.
   */
  PY2AND3,

  /**
   * Alias for {@code PY2}. Deprecated in Bazel; prefer {@code PY2}.
   *
   * <p><i>Deprecated meaning:</i> Indicates code that cannot be processed by 2to3.
   */
  PY2ONLY,

  /**
   * Deprecated alias for {@code PY3}.
   *
   * <p><i>Deprecated meaning:</i> Indicates code that cannot be processed by 3to2.
   */
  PY3ONLY,

  /**
   * Internal sentinel value used as the default value of the {@code python_version} and {@code
   * default_python_version} attributes.
   *
   * <p>This should not be referenced by the user. But since we can't actually hide it from Starlark
   * ({@code native.existing_rules()}) or bazel query, we give it the scary "_internal" prefix
   * instead.
   *
   * <p>The logical meaning of this value is the same as {@link
   * PythonOptions#getDefaultPythonVersion}.
   */
  _INTERNAL_SENTINEL;

  private static ImmutableList<String> convertToStrings(List<PythonVersion> values) {
    return values.stream()
        .map(Functions.toStringFunction())
        .collect(ImmutableList.toImmutableList());
  }

  /** Enum values representing a distinct Python version. */
  public static final ImmutableList<PythonVersion> TARGET_VALUES = ImmutableList.of(PY2, PY3);

  /** String names of enum values representing a distinct Python version. */
  public static final ImmutableList<String> TARGET_STRINGS = convertToStrings(TARGET_VALUES);

  /** Target values plus the sentinel value. */
  public static final ImmutableList<PythonVersion> TARGET_AND_SENTINEL_VALUES =
      ImmutableList.of(PY2, PY3, _INTERNAL_SENTINEL);

  /** String names of target values plus the sentinel value. */
  public static final ImmutableList<String> TARGET_AND_SENTINEL_STRINGS =
      convertToStrings(TARGET_AND_SENTINEL_VALUES);

  /** All values not including the sentinel. */
  public static final ImmutableList<PythonVersion> SRCS_VALUES =
      ImmutableList.of(PY2, PY3, PY2AND3, PY2ONLY, PY3ONLY);

  /** String names of all enum values not including the sentinel. */
  public static final ImmutableList<String> SRCS_STRINGS = convertToStrings(SRCS_VALUES);

  /** Enum values that do not imply running a transpiler to convert between versions. */
  public static final ImmutableList<PythonVersion> NON_CONVERSION_VALUES =
      ImmutableList.of(PY2AND3, PY2ONLY, PY3ONLY);

  /**
   * String names of enum values that do not imply running a transpiler to convert between versions.
   */
  public static final ImmutableList<String> NON_CONVERSION_STRINGS =
      convertToStrings(NON_CONVERSION_VALUES);

  public static final PythonVersion DEFAULT_SRCS_VALUE = PY2AND3;

  /** Returns whether or not this value is a distinct Python version. */
  public boolean isTargetValue() {
    return TARGET_VALUES.contains(this);
  }

  /**
   * Converts the string to a target {@code PythonVersion} value (case-sensitive).
   *
   * @throws IllegalArgumentException if the string is not "PY2" or "PY3".
   */
  public static PythonVersion parseTargetValue(String str) {
    if (!TARGET_STRINGS.contains(str)) {
      throw new IllegalArgumentException(
          String.format("'%s' is not a valid Python major version. Expected 'PY2' or 'PY3'.", str));
    }
    return PythonVersion.valueOf(str);
  }

  /**
   * Converts the string to a target or sentinel {@code PythonVersion} value (case-sensitive).
   *
   * @throws IllegalArgumentException if the string is not "PY2", "PY3", or "_INTERNAL_SENTINEL".
   */
  public static PythonVersion parseTargetOrSentinelValue(String str) {
    if (!TARGET_AND_SENTINEL_STRINGS.contains(str)) {
      // Use the same error message as for parseTargetValue, because the user shouldn't be aware of
      // the sentinel value.
      throw new IllegalArgumentException(
          String.format("'%s' is not a valid Python major version. Expected 'PY2' or 'PY3'.", str));
    }
    return PythonVersion.valueOf(str);
  }

  /**
   * Converts the string to a sources {@code PythonVersion} value (case-sensitive).
   *
   * @throws IllegalArgumentException if the string is not an enum name or is the sentinel value.
   */
  public static PythonVersion parseSrcsValue(String str) {
    if (!SRCS_STRINGS.contains(str)) {
      throw new IllegalArgumentException(
          String.format("'%s' is not a valid Python srcs_version value.", str));
    }
    return PythonVersion.valueOf(str);
  }
}

