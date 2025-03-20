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

import com.google.common.base.Joiner;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Starlark;

/** A class representing a Java method callable from Starlark with annotation. */
public final class StarlarkJavaMethodDoc extends StarlarkMethodDoc {
  private final String moduleName;
  private final String name;

  private boolean isOverloaded;

  public StarlarkJavaMethodDoc(
      String moduleName,
      Method javaMethod,
      StarlarkMethod annotation,
      StarlarkDocExpander expander) {
    super(javaMethod, annotation, expander);
    this.moduleName = moduleName;
    this.name = annotation.name();
  }

  @Override
  public String getName() {
    // Normally we refer to methods by their name, e.g. "foo" for method foo(arg1, arg2).
    // However, if a method is overloaded, the name is no longer unique, which forces us to append
    // the names of the method parameters in order to get a unique value.
    // In this case, the return value for the previous example would be "foo(arg1, arg2)".

    // We decided against ALWAYS returning the full name since we didn't want to pollute the
    // TOC of documentation pages too much. This comes at the cost of inconsistency and more
    // complex code.
    return isOverloaded ? getFullName() : name;
  }

  /**
   * Returns the full name of the method, consisting of <method name>(<name of first param>, <name
   * of second param>, ...).
   */
  private String getFullName() {
    List<String> paramNames = new ArrayList<>();
    for (Param param : annotation.parameters()) {
      paramNames.add(param.name());
    }
    return String.format("%s(%s)", name, Joiner.on(", ").join(paramNames));
  }

  @Override
  public String getShortName() {
    return name;
  }

  @Override
  public String getRawDocumentation() {
    String prefixWarning = "";
    if (!annotation.enableOnlyWithFlag().isEmpty()) {
      prefixWarning =
          "<b>Experimental</b>. This API is experimental and may change at any time. "
              + "Please do not depend on it. It may be enabled on an experimental basis by setting "
              + "<code>--"
              + annotation.enableOnlyWithFlag()
              + "</code> <br>";
    } else if (!annotation.disableWithFlag().isEmpty()) {
      prefixWarning =
          "<b>Deprecated</b>. This API is deprecated and will be removed soon. "
              + "Please do not depend on it. It is <i>disabled</i> with "
              + "<code>--"
              + annotation.disableWithFlag()
              + "</code>. Use this flag "
              + "to verify your code is compatible with its imminent removal. <br>";
    }
    return prefixWarning + annotation.doc();
  }

  @Override
  public String getSignature() {
    String objectDotExpressionPrefix = moduleName.isEmpty() ? "" : moduleName + ".";

    return getSignature(objectDotExpressionPrefix + name);
  }

  @Override
  public String getReturnType() {
    return Starlark.classTypeFromJava(javaMethod.getReturnType());
  }

  public void setOverloaded(boolean isOverloaded) {
    this.isOverloaded = isOverloaded;
  }
}
