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

  private static final String STARLARK_PREFIX = "_";

  private final String nativePrefix;
  private final boolean mustHaveStarlarkPrefix;

  /**
   * Creates a new instance and defines the prefixes for both Starlark and native.
   *
   * @param nativePrefix The prefix when converted to a native attribute name.
   * @param mustHaveStarlarkPrefix Whether the Starlark name must start with {@link
   *     AttributeValueSource#STARLARK_PREFIX}.
   */
  AttributeValueSource(String nativePrefix, boolean mustHaveStarlarkPrefix) {
    this.nativePrefix = nativePrefix;
    this.mustHaveStarlarkPrefix = mustHaveStarlarkPrefix;
  }

  /** Throws an {@link EvalException} if the given Starlark name is not valid for this type. */
  public void validateStarlarkName(String attrStarlarkName) throws EvalException {
    if (attrStarlarkName.isEmpty()) {
      throw new EvalException("Attribute name must not be empty.");
    }

    if (mustHaveStarlarkPrefix && !attrStarlarkName.startsWith(STARLARK_PREFIX)) {
      throw new EvalException(
          String.format(
              "When an attribute value is a function, the attribute must be private "
                  + "(i.e. start with '%s'). Found '%s'",
              STARLARK_PREFIX, attrStarlarkName));
    }
  }

  /**
   * Converts the given Starlark attribute name to a native attribute name for this type, or throws
   * an {@link EvalException} if the given Starlark name is not valid for this type.
   */
  public String convertToNativeName(String attrStarlarkName) throws EvalException {
    validateStarlarkName(attrStarlarkName);
    // No need to check for mustHaveStarlarkPrefix since this was already done in
    // validateStarlarkName().
    return attrStarlarkName.startsWith(STARLARK_PREFIX)
        ? nativePrefix + attrStarlarkName.substring(STARLARK_PREFIX.length())
        : attrStarlarkName;
  }
}
