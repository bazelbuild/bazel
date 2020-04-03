// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import com.google.common.collect.ImmutableList;
import com.google.errorprone.annotations.FormatMethod;
import com.google.errorprone.annotations.FormatString;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Element;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.type.TypeVariable;
import javax.lang.model.type.WildcardType;

class SerializationProcessorUtil {
  private SerializationProcessorUtil() {}

  // Sanitizes the type parameter. If it's a TypeVariable or WildcardType this will get the erasure.
  static TypeMirror sanitizeTypeParameter(TypeMirror type, ProcessingEnvironment env) {
    if (isVariableOrWildcardType(type)) {
      return env.getTypeUtils().erasure(type);
    }
    if (!(type instanceof DeclaredType)) {
      return type;
    }
    DeclaredType declaredType = (DeclaredType) type;
    for (TypeMirror typeMirror : declaredType.getTypeArguments()) {
      if (isVariableOrWildcardType(typeMirror)) {
        return env.getTypeUtils().erasure(type);
      }
    }
    return type;
  }

  static JavaFile writeGeneratedClassToFile(
      Element element, TypeSpec builtClass, ProcessingEnvironment env)
      throws SerializationProcessingFailedException {
    String packageName = env.getElementUtils().getPackageOf(element).getQualifiedName().toString();
    JavaFile file = JavaFile.builder(packageName, builtClass).build();
    try {
      file.writeTo(env.getFiler());
    } catch (IOException e) {
      throw new SerializationProcessingFailedException(
          element, "Failed to generate output file: %s", e.getMessage());
    }
    return file;
  }

  /**
   * Returns a class name generated from the given {@code element}.
   *
   * <p>For {@code Foo.Bar} this is {@code Foo_Bar_suffix}.
   */
  static String getGeneratedName(Element element, String suffix) {
    ImmutableList.Builder<String> classNamesBuilder = new ImmutableList.Builder<>();
    classNamesBuilder.add(suffix);
    do {
      classNamesBuilder.add(element.getSimpleName().toString());
      element = element.getEnclosingElement();
    } while (element instanceof TypeElement);
    return String.join("_", classNamesBuilder.build().reverse());
  }

  static boolean isVariableOrWildcardType(TypeMirror type) {
    return type instanceof TypeVariable || type instanceof WildcardType;
  }

  /** Indicates {@link AutoCodec}/{@link SerializationConstant} annotation processing failure. */
  static final class SerializationProcessingFailedException extends Exception {
    private final Element element;

    @FormatMethod
    SerializationProcessingFailedException(
        Element element, @FormatString String fmt, Object... args) {
      super(String.format(fmt, args));
      this.element = element;
    }

    Element getElement() {
      return element;
    }
  }
}
