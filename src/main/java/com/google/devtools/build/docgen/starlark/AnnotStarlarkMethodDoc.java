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

import com.google.common.collect.ImmutableList;
import java.lang.reflect.Method;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/**
 * An abstract class containing documentation for a {@link StarlarkMethod}-annotated Java method
 * callable from Starlark.
 */
public abstract class AnnotStarlarkMethodDoc extends MemberDoc {
  protected final Method javaMethod;
  protected final StarlarkMethod annotation;
  protected final ImmutableList<AnnotParamDoc> params;

  public AnnotStarlarkMethodDoc(
      Method method, StarlarkMethod annotation, StarlarkDocExpander expander) {
    super(expander);
    this.javaMethod = method;
    this.annotation = annotation;
    this.params = determineParams();
  }

  @Override
  public final boolean documented() {
    return annotation.documented();
  }

  @Override
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

  @Override
  public final boolean isCallable() {
    return !annotation.structField();
  }

  /** Returns a list containing the documentation for each of the method's parameters. */
  @Override
  public final ImmutableList<AnnotParamDoc> getParams() {
    return params;
  }

  private ImmutableList<AnnotParamDoc> determineParams() {
    ImmutableList.Builder<AnnotParamDoc> paramsBuilder = ImmutableList.builder();
    for (int i = getStartIndexForParams(); i < annotation.parameters().length; i++) {
      Param param = annotation.parameters()[i];
      if (param.documented()) {
        ParamDoc.Kind kind = ParamDoc.Kind.ORDINARY;
        if (!param.named()) {
          kind = ParamDoc.Kind.POSITIONAL_ONLY;
        } else if (!param.positional()) {
          kind = ParamDoc.Kind.KEYWORD_ONLY;
        }
        paramsBuilder.add(new AnnotParamDoc(this, param, expander, kind, i));
      }
    }
    if (!annotation.extraPositionals().name().isEmpty()) {
      paramsBuilder.add(
          new AnnotParamDoc(
              this,
              annotation.extraPositionals(),
              expander,
              ParamDoc.Kind.VARARGS,
              /* paramIndex= */ -1));
    }
    if (!annotation.extraKeywords().name().isEmpty()) {
      paramsBuilder.add(
          new AnnotParamDoc(
              this,
              annotation.extraKeywords(),
              expander,
              ParamDoc.Kind.KWARGS,
              /* paramIndex= */ -1));
    }
    return paramsBuilder.build();
  }

  protected String getSignature(String fullyQualifiedMethodName) {
    String args = isCallable() ? "(" + getParameterString() + ")" : "";

    return String.format(
        "%s %s%s", getTypeAnchor(javaMethod.getReturnType()), fullyQualifiedMethodName, args);
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
