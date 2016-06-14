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

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BaseFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * A class representing a Skylark built-in object or method.
 */
public final class SkylarkBuiltinMethodDoc extends SkylarkMethodDoc {
  private final SkylarkModuleDoc module;
  private final SkylarkSignature annotation;
  private final Class<?> fieldClass;
  private List<SkylarkParamDoc> params;

  public SkylarkBuiltinMethodDoc(SkylarkModuleDoc module, SkylarkSignature annotation,
      Class<?> fieldClass) {
    this.module = module;
    this.annotation = annotation;
    this.fieldClass = fieldClass;
    this.params = new ArrayList<>();
    processParams();
  }

  public SkylarkSignature getAnnotation() {
    return annotation;
  }

  @Override
  public boolean documented() {
    return annotation.documented();
  }

  @Override
  public String getName() {
    return annotation.name();
  }

  @Override
  public String getDocumentation() {
    return annotation.doc();
  }

  /**
   * Returns a string representing the method signature with links to the types if
   * available.
   *
   * <p>If the built-in method is a function, the construct the method signature. Otherwise,
   * return a string containing the return type of the method.
   */
  @Override
  public String getSignature() {
    if (BaseFunction.class.isAssignableFrom(fieldClass)) {
      return getSignature(module.getName(), annotation);
    }
    if (!annotation.returnType().equals(Object.class)) {
      return getTypeAnchor(annotation.returnType());
    }
    return "";
  }

  @Override
  public List<SkylarkParamDoc> getParams() {
    return params;
  }

  private void processParams() {
    processParams(adjustedParameters(annotation));
    if (!annotation.extraPositionals().name().isEmpty()) {
      this.params.add(new SkylarkParamDoc(this, annotation.extraPositionals()));
    }
    if (!annotation.extraKeywords().name().isEmpty()) {
      this.params.add(new SkylarkParamDoc(this, annotation.extraKeywords()));
    }
  }

  private void processParams(Param[] params) {
    for (Param param : params) {
      this.params.add(new SkylarkParamDoc(this, param));
    }
  }
}
