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

/** Documentation for a function parameter. */
public abstract class ParamDoc extends StarlarkDoc {
  /** Represents the param kind, whether it's a normal param or *arg or **kwargs. */
  public static enum Kind {
    NORMAL,
    // TODO: https://github.com/bazelbuild/stardoc/issues/225 - NORMAL needs to be split into
    //   NORMAL and KEYWORD_ONLY, since EXTRA_KEYWORDS (or a `*` separator) go before keyword-only
    //   params, not necessarily immediately before kwargs.
    EXTRA_POSITIONALS,
    EXTRA_KEYWORDS,
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
