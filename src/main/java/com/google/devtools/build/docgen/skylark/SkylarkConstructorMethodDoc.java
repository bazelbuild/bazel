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
package com.google.devtools.build.docgen.skylark;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.syntax.EvalUtils;
import java.lang.reflect.Method;
import java.util.List;

/**
 * A class representing a Java method callable from Skylark which constructs a type of
 * skylark object. Such a method is annotated with {@link SkylarkConstructor}, and has special
 * handling.
 */
public final class SkylarkConstructorMethodDoc extends SkylarkMethodDoc {
  private final String fullyQualifiedName;
  private final Method method;
  private final SkylarkCallable callable;
  private final ImmutableList<SkylarkParamDoc> params;

  public SkylarkConstructorMethodDoc(
      String fullyQualifiedName, Method method, SkylarkCallable callable) {
    this.fullyQualifiedName = fullyQualifiedName;
    this.method = method;
    this.callable = callable;
    this.params =
        SkylarkDocUtils.determineParams(
            this,
            withoutSelfParam(callable, method),
            callable.extraPositionals(),
            callable.extraKeywords());
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
    return SkylarkDocUtils.substituteVariables(callable.doc());
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
    return EvalUtils.getDataTypeNameFromClass(method.getReturnType());
  }

  @Override
  public List<SkylarkParamDoc> getParams() {
    return params;
  }
}
