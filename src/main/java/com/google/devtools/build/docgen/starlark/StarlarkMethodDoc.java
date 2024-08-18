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
import com.google.common.collect.ImmutableList;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** An abstract class containing documentation for a Starlark method. */
public abstract class StarlarkMethodDoc extends StarlarkDoc {
  protected final Method javaMethod;
  protected final StarlarkMethod annotation;
  protected final ImmutableList<StarlarkParamDoc> params;

  public StarlarkMethodDoc(Method method, StarlarkMethod annotation, StarlarkDocExpander expander) {
    super(expander);
    this.javaMethod = method;
    this.annotation = annotation;
    this.params = determineParams();
  }

  /** Returns whether the Starlark method is documented. */
  public final boolean documented() {
    return annotation.documented();
  }

  /**
   * Returns a string containing additional documentation about the method's return value.
   *
   * <p>Returns an empty string by default.
   */
  public final String getReturnTypeExtraMessage() {
    if (annotation.allowReturnNones()) {
      return " May return <code>None</code>.\n";
    }
    return "";
  }

  /** Returns the annotated Java method. */
  public final Method getMethod() {
    return javaMethod;
  }

  /** Returns a string containing a name for the method's return type. */
  public abstract String getReturnType();

  /**
   * Returns whether a method can be called as a function.
   *
   * <p>E.g. ctx.label is not callable.
   */
  public final boolean isCallable() {
    return !annotation.structField();
  }

  /**
   * Returns a string containing the method's name. GetName() returns the complete signature in case
   * of overloaded methods. This is used to extract only the name of the method.
   *
   * <p>E.g. ctx.new_file is overloaded. In this case getName() returns "new_file(filename)", while
   * getShortName() returns only "new_file".
   */
  public String getShortName() {
    return getName();
  }

  /** Returns a list containing the documentation for each of the method's parameters. */
  public final ImmutableList<StarlarkParamDoc> getParams() {
    return params;
  }

  private String getParameterString() {
    List<String> argList = new ArrayList<>();

    boolean named = false;
    for (int i = getStartIndexForParams(); i < annotation.parameters().length; i++) {
      Param param = annotation.parameters()[i];
      if (!param.documented()) {
        continue;
      }
      if (param.named() && !param.positional() && !named) {
        named = true;
        if (!argList.isEmpty()) {
          argList.add("*");
        }
      }
      argList.add(formatParameter(param));
    }
    if (!annotation.extraPositionals().name().isEmpty()) {
      argList.add("*" + annotation.extraPositionals().name());
    }
    if (!annotation.extraKeywords().name().isEmpty()) {
      argList.add("**" + annotation.extraKeywords().name());
    }
    return Joiner.on(", ").join(argList);
  }

  private ImmutableList<StarlarkParamDoc> determineParams() {
    ImmutableList.Builder<StarlarkParamDoc> paramsBuilder = ImmutableList.builder();
    for (int i = getStartIndexForParams(); i < annotation.parameters().length; i++) {
      Param param = annotation.parameters()[i];
      if (param.documented()) {
        paramsBuilder.add(
            new StarlarkParamDoc(this, param, expander, StarlarkParamDoc.Kind.NORMAL, i));
      }
    }
    if (!annotation.extraPositionals().name().isEmpty()) {
      paramsBuilder.add(
          new StarlarkParamDoc(
              this,
              annotation.extraPositionals(),
              expander,
              StarlarkParamDoc.Kind.EXTRA_POSITIONALS,
              /* paramIndex= */ -1));
    }
    if (!annotation.extraKeywords().name().isEmpty()) {
      paramsBuilder.add(
          new StarlarkParamDoc(
              this,
              annotation.extraKeywords(),
              expander,
              StarlarkParamDoc.Kind.EXTRA_KEYWORDS,
              /* paramIndex= */ -1));
    }
    return paramsBuilder.build();
  }

  /**
   * Returns a string representing the method signature of the Starlark method, which contains HTML
   * links to the documentation of parameter types if available.
   */
  public abstract String getSignature();

  protected String getSignature(String fullyQualifiedMethodName) {
    String args = isCallable() ? "(" + getParameterString() + ")" : "";

    return String.format(
        "%s %s%s", getTypeAnchor(javaMethod.getReturnType()), fullyQualifiedMethodName, args);
  }

  private String formatParameter(Param param) {
    String defaultValue = param.defaultValue();
    String name = param.name();
    if (defaultValue == null || !defaultValue.isEmpty()) {
      return String.format("%s=%s", name, defaultValue == null ? "&hellip;" : defaultValue);
    } else {
      return name;
    }
  }

  /**
   * Returns the index to start at when iterating through the parameters of the method annotation.
   * This is not always 0 because of the "self" param for the "string" module.
   */
  private int getStartIndexForParams() {
    Param[] params = annotation.parameters();
    if (params.length > 0) {
      StarlarkBuiltin module = javaMethod.getDeclaringClass().getAnnotation(StarlarkBuiltin.class);
      if (module != null && module.name().equals("string")) {
        // Skip the self parameter, which is the first mandatory
        // positional parameter in each method of the "string" module.
        return 1;
      }
    }
    return 0;
  }
}
