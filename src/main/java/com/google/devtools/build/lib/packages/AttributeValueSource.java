// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.syntax.EvalException;

/**
 * An enum that represents different types of rule attributes, based on where their values come
 * from.
 */
public enum AttributeValueSource {
  COMPUTED_DEFAULT("$", true),
  LATE_BOUND(":", true),
  DIRECT("$", false);

  private static final String SKYLARK_PREFIX = "_";

  private final String nativePrefix;
  private final boolean mustHaveSkylarkPrefix;

  /**
   * Creates a new instance and defines the prefixes for both Skylark and native.
   *
   * @param nativePrefix The prefix when converted to a native attribute name.
   * @param mustHaveSkylarkPrefix Whether the Skylark name must start with {@link
   *     AttributeValueSource#SKYLARK_PREFIX}.
   */
  AttributeValueSource(String nativePrefix, boolean mustHaveSkylarkPrefix) {
    this.nativePrefix = nativePrefix;
    this.mustHaveSkylarkPrefix = mustHaveSkylarkPrefix;
  }

  /** Throws an {@link EvalException} if the given Skylark name is not valid for this type. */
  public void validateSkylarkName(String attrSkylarkName) throws EvalException {
    if (attrSkylarkName.isEmpty()) {
      throw new EvalException(null, "Attribute name must not be empty.");
    }

    if (mustHaveSkylarkPrefix && !attrSkylarkName.startsWith(SKYLARK_PREFIX)) {
      throw new EvalException(
          null,
          String.format(
              "When an attribute value is a function, the attribute must be private "
                  + "(i.e. start with '%s'). Found '%s'",
              SKYLARK_PREFIX, attrSkylarkName));
    }
  }

  /**
   * Converts the given Skylark attribute name to a native attribute name for this type, or throws
   * an {@link EvalException} if the given Skylark name is not valid for this type.
   */
  public String convertToNativeName(String attrSkylarkName) throws EvalException {
    validateSkylarkName(attrSkylarkName);
    // No need to check for mustHaveSkylarkPrefix since this was already done in
    // validateSkylarkName().
    return attrSkylarkName.startsWith(SKYLARK_PREFIX)
        ? nativePrefix + attrSkylarkName.substring(SKYLARK_PREFIX.length())
        : attrSkylarkName;
  }
}
