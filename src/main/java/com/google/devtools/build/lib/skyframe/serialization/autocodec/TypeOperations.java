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
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.util.Set;
import javax.annotation.Nullable;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Element;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.ArrayType;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.type.TypeVariable;
import javax.lang.model.type.WildcardType;

/** Common {@link TypeMirror} and {@link Element} operations. */
final class TypeOperations {
  static TypeMirror getTypeMirror(Class<?> clazz, ProcessingEnvironment env) {
    return env.getElementUtils().getTypeElement(clazz.getCanonicalName()).asType();
  }

  /**
   * Sanitizes the type parameter.
   *
   * <p>If it's a {@link TypeVariable} or {@link WildcardType} returns the erasure.
   */
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
      throws SerializationProcessingException {
    String packageName = env.getElementUtils().getPackageOf(element).getQualifiedName().toString();
    JavaFile file = JavaFile.builder(packageName, builtClass).build();
    try {
      file.writeTo(env.getFiler());
    } catch (IOException e) {
      throw new SerializationProcessingException(
          element, "Failed to generate output file: %s", e.getMessage());
    }
    return file;
  }

  /**
   * Generates a name from the given {@code element} and {@code suffix}.
   *
   * <p>For a class {@code Foo.Bar} this is {@code Foo_Bar_suffix}. For a variable {@code
   * Foo.Bar.baz}, this is {@code Foo_Bar_baz_suffix}.
   */
  static String getGeneratedName(Element element, String suffix) {
    ImmutableList.Builder<String> nameComponents = new ImmutableList.Builder<>();
    nameComponents.add(suffix);
    do {
      nameComponents.add(element.getSimpleName().toString());
      element = element.getEnclosingElement();
    } while (element instanceof TypeElement);
    return String.join("_", nameComponents.build().reverse());
  }

  static boolean isVariableOrWildcardType(TypeMirror type) {
    return type instanceof TypeVariable || type instanceof WildcardType;
  }

  /** True when {@code type} has the same type as {@code clazz}. */
  static boolean matchesType(TypeMirror type, Class<?> clazz, ProcessingEnvironment env) {
    return env.getTypeUtils().isSameType(type, getTypeMirror(clazz, env));
  }

  /**
   * Returns the erased type.
   *
   * <p>This is {@link TypeName} rather than {@link TypeMirror} because it is more compatible with
   * Javapoet's API methods.
   */
  static TypeName getErasure(TypeMirror type, ProcessingEnvironment env) {
    return TypeName.get(getErasureAsMirror(type, env));
  }

  static TypeName getErasure(TypeElement type, ProcessingEnvironment env) {
    return getErasure(type.asType(), env);
  }

  static TypeMirror getErasureAsMirror(TypeElement type, ProcessingEnvironment env) {
    return getErasureAsMirror(type.asType(), env);
  }

  static TypeMirror getErasureAsMirror(TypeMirror type, ProcessingEnvironment env) {
    return env.getTypeUtils().erasure(type);
  }

  static boolean isSerializableField(VariableElement variable) {
    Set<Modifier> modifiers = variable.getModifiers();
    return !modifiers.contains(Modifier.STATIC) && !modifiers.contains(Modifier.TRANSIENT);
  }

  @Nullable
  static TypeElement getSuperclass(TypeElement type) {
    TypeMirror mirror = type.getSuperclass();
    if (!(mirror instanceof DeclaredType)) {
      // `type` represents Object or some interface.
      return null;
    }
    // `DeclaredType.asElement` can return a `TypeParameterElement` instance if `mirror` is a
    // generic type parameter, which isn't possible here.
    return (TypeElement) ((DeclaredType) mirror).asElement();
  }

  static TypeMirror resolveBaseArrayComponentType(TypeMirror type) {
    if (!type.getKind().equals(TypeKind.ARRAY)) {
      return type;
    }
    ArrayType arrayType = (ArrayType) type;
    TypeMirror componentType = arrayType.getComponentType();
    return resolveBaseArrayComponentType(componentType);
  }

  private TypeOperations() {}
}
