// Copyright 2018 The Bazel Authors. All rights reserved.
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
import java.util.List;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Starlark;

/**
 * A class representing a Java method callable from Starlark which constructs a type of Starlark
 * object. Such a method is annotated with {@link StarlarkConstructor}, and has special handling.
 */
public final class StarlarkConstructorMethodDoc extends StarlarkMethodDoc {
  private final String fullyQualifiedName;
  private final Method method;
  private final StarlarkMethod callable;
  private final ImmutableList<StarlarkParamDoc> params;

  public StarlarkConstructorMethodDoc(
      String fullyQualifiedName,
      Method method,
      StarlarkMethod callable,
      StarlarkDocExpander expander) {
    super(expander);
    this.fullyQualifiedName = fullyQualifiedName;
    this.method = method;
    this.callable = callable;
    this.params =
        StarlarkDocUtils.determineParams(
            this,
            withoutSelfParam(callable, method),
            callable.extraPositionals(),
            callable.extraKeywords(),
            expander);
  }

  @Override
  public Method getMethod() {
    return method;
  }

  @Override
  public boolean documented() {
    return callable.documented();
  }

  @Override
  public String getName() {
    return fullyQualifiedName;
  }

  @Override
  public String getRawDocumentation() {
    return callable.doc();
  }

  @Override
  public String getSignature() {
    return getSignature(fullyQualifiedName, method);
  }

  @Override
  public String getReturnTypeExtraMessage() {
    if (callable.allowReturnNones()) {
      return " May return <code>None</code>.\n";
    }
    return "";
  }

  @Override
  public String getReturnType() {
    return Starlark.classType(method.getReturnType());
  }

  @Override
  public List<StarlarkParamDoc> getParams() {
    return params;
  }

  @Override
  public String toString() {
    return String.format(
        "StarlarkConstructorMethodDoc{fullyQualifiedName=%s method=%s callable=%s}",
        fullyQualifiedName, method, formatCallable());
  }

  private String formatCallable() {
    return String.format(
        "StarlarkMethod{name=%s selfCall=%s structField=%s doc=%s}",
        callable.name(), callable.selfCall(), callable.structField(), callable.doc());
  }
}
