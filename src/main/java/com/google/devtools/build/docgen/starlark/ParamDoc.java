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
package com.google.devtools.build.docgen.starlark;

import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole;

/** Documentation for a function parameter. */
public abstract class ParamDoc extends StarlarkDoc {
  /**
   * Represents the param kind, e.g. whether it's an ordinary parameter, keyword-only, *args,
   * **kwargs, etc.
   */
  // Keep in sync with FunctionParamRole in stardoc_output.proto.
  public static enum Kind {
    /** An ordinary parameter which may be used as a positional or by keyword. */
    ORDINARY,
    /**
     * A positional-only parameter; such parameters cannot be defined in pure Starlark code, but
     * exist in some natively-defined functions.
     */
    POSITIONAL_ONLY,
    /**
     * A keyword-only parameter, i.e. a non-vararg/kwarg parameter that follows `*` or `*args` in
     * the function's declaration.
     */
    KEYWORD_ONLY,
    /** Residual varargs, typically `*args` in the function's declaration. */
    VARARGS,
    /** Residual keyword arguments, typically `**kwargs` in the function's declaration. */
    KWARGS;

    public static Kind fromProto(FunctionParamRole role) {
      return switch (role) {
        case PARAM_ROLE_ORDINARY -> ORDINARY;
        case PARAM_ROLE_POSITIONAL_ONLY -> POSITIONAL_ONLY;
        case PARAM_ROLE_KEYWORD_ONLY -> KEYWORD_ONLY;
        case PARAM_ROLE_VARARGS -> VARARGS;
        case PARAM_ROLE_KWARGS -> KWARGS;
        default -> throw new IllegalArgumentException("Unknown param role: " + role);
      };
    }
  }

  protected final Kind kind;

  public ParamDoc(StarlarkDocExpander expander, Kind kind) {
    super(expander);
    this.kind = kind;
  }

  /**
   * Returns the string representing the type of this parameter with the link to the documentation
   * for the type if available.
   *
   * <p>If the parameter type is unspecified (e.g. {@link Object} for a Java-defined method), then
   * returns the empty string. If the parameter type is not a generic, then this method returns a
   * string representing the type name with a link to the documentation for the type if available.
   * If the parameter type is a generic, then this method returns a string "CONTAINER of TYPE" (with
   * HTML link markup).
   */
  public abstract String getType();

  public Kind getKind() {
    return kind;
  }

  public abstract String getDefaultValue();
}
