// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Element;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.MirroredTypesException;
import javax.lang.model.type.TypeMirror;

/** Static utilities for AutoCodec processors. */
class AutoCodecUtil {
  // Synthesized classes will have `_AutoCodec` suffix added.
  private static final String GENERATED_CLASS_NAME_SUFFIX = "AutoCodec";
  static final Class<AutoCodec> ANNOTATION = AutoCodec.class;

  /**
   * Initializes a builder for a class of the appropriate type.
   *
   * @param encodedType type being serialized
   */
  static TypeSpec.Builder initializeCodecClassBuilder(
      TypeElement encodedType, ProcessingEnvironment env) {
    TypeSpec.Builder builder = TypeSpec.classBuilder(getCodecName(encodedType));
    return builder.addSuperinterface(
        ParameterizedTypeName.get(
            ClassName.get(ObjectCodec.class),
            TypeName.get(env.getTypeUtils().erasure(encodedType.asType()))));
  }

  static MethodSpec.Builder initializeGetEncodedClassMethod(
      TypeElement encodedType, ProcessingEnvironment env) {
    return MethodSpec.methodBuilder("getEncodedClass")
        .addModifiers(Modifier.PUBLIC)
        .addAnnotation(Override.class)
        .returns(
            ParameterizedTypeName.get(
                ClassName.get(Class.class),
                TypeName.get(env.getTypeUtils().erasure(encodedType.asType()))));
  }

  /**
   * Initializes the deserialize method.
   *
   * @param encodedType type being serialized
   */
  static MethodSpec.Builder initializeSerializeMethodBuilder(
      TypeElement encodedType, AutoCodec annotation, ProcessingEnvironment env) {
    MethodSpec.Builder builder =
        MethodSpec.methodBuilder("serialize")
            .addModifiers(Modifier.PUBLIC)
            .returns(void.class)
            .addAnnotation(Override.class)
            .addException(SerializationException.class)
            .addException(IOException.class)
            .addParameter(SerializationContext.class, "context")
            .addParameter(TypeName.get(env.getTypeUtils().erasure(encodedType.asType())), "input")
            .addParameter(CodedOutputStream.class, "codedOut");
    if (annotation.checkClassExplicitlyAllowed()) {
      builder.addStatement("context.checkClassExplicitlyAllowed(getEncodedClass(), input)");
    }
    List<? extends TypeMirror> explicitlyAllowedClasses;
    try {
      explicitlyAllowedClasses =
          Arrays.stream(annotation.explicitlyAllowClass())
              .map((clazz) -> getType(clazz, env))
              .collect(Collectors.toList());
    } catch (MirroredTypesException e) {
      explicitlyAllowedClasses = e.getTypeMirrors();
    }
    for (TypeMirror explicitlyAllowedClass : explicitlyAllowedClasses) {
      builder.addStatement("context.addExplicitlyAllowedClass($T.class)", explicitlyAllowedClass);
    }
    return builder;
  }

  /**
   * Initializes the deserialize method.
   *
   * @param encodedType type being serialized
   */
  static MethodSpec.Builder initializeDeserializeMethodBuilder(
      TypeElement encodedType, ProcessingEnvironment env) {
    MethodSpec.Builder builder =
        MethodSpec.methodBuilder("deserialize")
            .addModifiers(Modifier.PUBLIC)
            .returns(TypeName.get(env.getTypeUtils().erasure(encodedType.asType())))
            .addAnnotation(Override.class)
            .addException(SerializationException.class)
            .addException(IOException.class)
            .addParameter(DeserializationContext.class, "context")
            .addParameter(CodedInputStream.class, "codedIn");
    return builder;
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
    return classNamesBuilder.build().reverse().stream().collect(Collectors.joining("_"));
  }

  /**
   * Name of the generated codec class.
   *
   * <p>For {@code Foo.Bar} this is {@code Foo_Bar_AutoCodec}.
   */
  private static String getCodecName(Element element) {
    return getGeneratedName(element, GENERATED_CLASS_NAME_SUFFIX);
  }

  static TypeMirror getType(Class<?> clazz, ProcessingEnvironment env) {
    return env.getElementUtils().getTypeElement((clazz.getCanonicalName())).asType();
  }

  static boolean isSubType(TypeMirror type, Class<?> clazz, ProcessingEnvironment env) {
    return env.getTypeUtils().isSubtype(type, getType(clazz, env));
  }
}
