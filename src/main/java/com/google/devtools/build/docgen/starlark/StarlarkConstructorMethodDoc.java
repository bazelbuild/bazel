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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkDeprecated;
import com.google.devtools.build.lib.syntax.Starlark;
import java.lang.reflect.Method;
import java.util.List;

/**
 * A class representing a Java method callable from Starlark which constructs a type of Starlark
 * object. Such a method is annotated with {@link SkylarkConstructor}, and has special handling.
 */
public final class StarlarkConstructorMethodDoc extends StarlarkMethodDoc {
  private final String fullyQualifiedName;
  private final Method method;
  private final SkylarkCallable callable;
  private final ImmutableList<StarlarkParamDoc> params;
  // TODO(cparsons): Move to superclass when SkylarkBuiltinMethodDoc is removed.
  private final boolean deprecated;

  public StarlarkConstructorMethodDoc(
      String fullyQualifiedName, Method method, SkylarkCallable callable) {
    this.fullyQualifiedName = fullyQualifiedName;
    this.method = method;
    this.callable = callable;
    this.params =
        StarlarkDocUtils.determineParams(
            this,
            withoutSelfParam(callable, method),
            callable.extraPositionals(),
            callable.extraKeywords());
    this.deprecated = method.isAnnotationPresent(StarlarkDeprecated.class);
  }

  @Override
  public boolean isDeprecated() {
    return deprecated;
  }

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
  public String getDocumentation() {
    return StarlarkDocUtils.substituteVariables(callable.doc());
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
}
