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

import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;

/**
 * A class containing the documentation for a parameter of a {@link
 * net.starlark.java.annot.StarlarkMethod}-annotated Java method callable from Starlark.
 */
public final class AnnotParamDoc extends ParamDoc {
  private final AnnotStarlarkMethodDoc method;
  private final Param param;
  private final int paramIndex;

  public AnnotParamDoc(
      AnnotStarlarkMethodDoc method,
      Param param,
      StarlarkDocExpander expander,
      Kind kind,
      int paramIndex) {
    super(expander, kind);
    this.method = method;
    this.param = param;
    this.paramIndex = paramIndex;
  }

  @Override
  public String getType() {
    StringBuilder sb = new StringBuilder();
    if (param.allowedTypes().length == 0) {
      // There is no `allowedTypes` field; we need to figure it out from the Java type.
      if (kind == Kind.ORDINARY || kind == Kind.POSITIONAL_ONLY || kind == Kind.KEYWORD_ONLY) {
        // Only deal with normal args for now; unclear what we could do for varargs.
        Class<?> type = method.getMethod().getParameterTypes()[paramIndex];
        if (type != Object.class) {
          sb.append(getTypeAnchor(type));
        }
      }
    } else {
      for (int i = 0; i < param.allowedTypes().length; i++) {
        ParamType paramType = param.allowedTypes()[i];
        // TODO(adonovan): make generic1 an array.
        if (paramType.generic1() == Object.class) {
          sb.append(getTypeAnchor(paramType.type()));
        } else {
          sb.append(getTypeAnchor(paramType.type(), paramType.generic1()));
        }
        if (i < param.allowedTypes().length - 1) {
          sb.append("; or ");
        }
      }
    }
    return sb.toString();
  }

  public AnnotStarlarkMethodDoc getMethod() {
    return method;
  }

  @Override
  public String getName() {
    return param.name();
  }

  @Override
  public String getDefaultValue() {
    return param.defaultValue();
  }

  @Override
  public String getRawDocumentation() {
    String prefixWarning = "";
    if (!param.enableOnlyWithFlag().isEmpty()) {
      prefixWarning =
          "<b>Experimental</b>. This parameter is experimental and may change at any "
              + "time. Please do not depend on it. It may be enabled on an experimental basis by "
              + "setting <code>--"
              + param.enableOnlyWithFlag().substring(1)
              + "</code> <br>";
    } else if (!param.disableWithFlag().isEmpty()) {
      prefixWarning =
          "<b>Deprecated</b>. This parameter is deprecated and will be removed soon. "
              + "Please do not depend on it. It is <i>disabled</i> with "
              + "<code>--"
              + param.disableWithFlag().substring(1)
              + "</code>. Use this flag "
              + "to verify your code is compatible with its imminent removal. <br>";
    }
    return prefixWarning + param.doc();
  }
}
