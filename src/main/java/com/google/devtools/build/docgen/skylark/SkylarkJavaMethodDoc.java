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
package com.google.devtools.build.docgen.skylark;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.util.StringUtilities;
import java.lang.reflect.Method;
import java.util.List;

/**
 * A class representing a Java method callable from Skylark with annotation.
 */
public final class SkylarkJavaMethodDoc extends SkylarkMethodDoc {
  private final SkylarkModuleDoc module;
  private final String name;
  private final Method method;
  private final SkylarkCallable callable;
  private final ImmutableList<SkylarkParamDoc> params;

  public SkylarkJavaMethodDoc(SkylarkModuleDoc module, Method method,
      SkylarkCallable callable) {
    this.module = module;
    this.name = callable.name().isEmpty()
        ? StringUtilities.toPythonStyleFunctionName(method.getName())
        : callable.name();
    this.method = method;
    this.callable = callable;
    ImmutableList.Builder<SkylarkParamDoc> paramsBuilder = ImmutableList.builder();
    for (Param param : callable.parameters()) {
      paramsBuilder.add(new SkylarkParamDoc(this, param));
    }
    this.params = paramsBuilder.build();
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
    return name;
  }

  @Override
  public String getDocumentation() {
    return callable.doc();
  }

  @Override
  public String getSignature() {
    return getSignature(module.getName(), name, method);
  }

  @Override
  public String getReturnTypeExtraMessage() {
    if (callable.allowReturnNones()) {
      return " May return <code>None</code>.\n";
    }
    return "";
  }

  @Override
  public List<SkylarkParamDoc> getParams() {
    return params;
  }
}
