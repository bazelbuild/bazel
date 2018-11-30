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

import static com.google.common.collect.Iterables.transform;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import java.util.Arrays;

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
  PY3ONLY;

  private static Iterable<String> convertToStrings(PythonVersion[] values) {
    return transform(ImmutableList.copyOf(values), Functions.toStringFunction());
  }

  private static final PythonVersion[] allValues =
      new PythonVersion[] {PY2, PY3, PY2AND3, PY2ONLY, PY3ONLY};

  private static final Iterable<String> allStrings = convertToStrings(allValues);

  private static final PythonVersion[] targetValues = new PythonVersion[] {PY2, PY3};

  private static final Iterable<String> targetStrings = convertToStrings(targetValues);

  private static final PythonVersion[] nonConversionValues =
      new PythonVersion[] {PY2AND3, PY2ONLY, PY3ONLY};

  private static final Iterable<String> nonConversionStrings =
      convertToStrings(nonConversionValues);

  private static final PythonVersion DEFAULT_TARGET_VALUE = PY2;

  private static final PythonVersion DEFAULT_SRCS_VALUE = PY2AND3;

  /** Returns all values as a new array. */
  public static PythonVersion[] getAllValues() {
    return Arrays.copyOf(allValues, allValues.length);
  }

  /** Returns an iterable of all values as strings. */
  public static Iterable<String> getAllStrings() {
    return allStrings;
  }

  /** Returns all values representing a specific version, as a new array. */
  public static PythonVersion[] getTargetValues() {
    return Arrays.copyOf(targetValues, targetValues.length);
  }

  /** Returns an iterable of all values representing a specific version, as strings. */
  public static Iterable<String> getTargetStrings() {
    return targetStrings;
  }

  /**
   * Returns all values that do not imply running a transpiler to convert between versions, as a new
   * array.
   */
  public static PythonVersion[] getNonConversionValues() {
    return Arrays.copyOf(nonConversionValues, nonConversionValues.length);
  }

  /**
   * Returns all values that do not imply running a transpiler to convert between versions, as
   * strings.
   */
  public static Iterable<String> getNonConversionStrings() {
    return nonConversionStrings;
  }

  /** Returns the Python version to use if not otherwise specified by a flag or attribute. */
  public static PythonVersion getDefaultTargetValue() {
    return DEFAULT_TARGET_VALUE;
  }

  /**
   * Returns the level of source compatibility assumed if not otherwise specified by an attribute.
   */
  public static PythonVersion getDefaultSrcsValue() {
    return DEFAULT_SRCS_VALUE;
  }

  // TODO(brandjon): Refactor this into parseTargetValue and parseSourcesValue methods. Throw
  // IllegalArgumentException on bad values instead of returning null, and modify callers to
  // tolerate the exception.
  /**
   * Converts the string to PythonVersion, if it is one of the allowed values. Returns null if the
   * input is not valid.
   */
  public static PythonVersion parse(String str, PythonVersion... allowed) {
    if (str == null) {
      return null;
    }
    try {
      PythonVersion version = PythonVersion.valueOf(str);
      if (Arrays.asList(allowed).contains(version)) {
        return version;
      }
      return null;
    } catch (IllegalArgumentException e) {
      return null;
    }
  }
}

