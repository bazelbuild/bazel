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
import com.google.devtools.build.lib.skyframe.serialization.InjectingObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import javax.lang.model.element.Element;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;

/** Static utilities for AutoCodec processors. */
class AutoCodecUtil {
  // Synthesized classes will have `_AutoCodec` suffix added.
  public static final String GENERATED_CLASS_NAME_SUFFIX = "AutoCodec";
  static final Class<AutoCodec> ANNOTATION = AutoCodec.class;

  /**
   * Initializes a builder for a class implementing {@link ObjectCodec}.
   *
   * @param encodedType type being serialized
   */
  static TypeSpec.Builder initializeCodecClassBuilder(TypeElement encodedType) {
    return initializeCodecClassBuilder(encodedType, null);
  }

  /**
   * Initializes a builder for a class of the appropriate type.
   *
   * <p>If the dependency is non-null, then the type is {@link InjectingObjectCodec} otherwise
   * {@link ObjectCodec}.
   *
   * @param encodedType type being serialized
   * @param dependency type being injected or null
   */
  static TypeSpec.Builder initializeCodecClassBuilder(
      TypeElement encodedType, @Nullable TypeElement dependency) {
    TypeSpec.Builder builder = TypeSpec.classBuilder(getCodecName(encodedType));
    if (dependency == null) {
      return builder.addSuperinterface(
          ParameterizedTypeName.get(
              ClassName.get(ObjectCodec.class), TypeName.get(encodedType.asType())));
    }
    return builder.addSuperinterface(
        ParameterizedTypeName.get(
            ClassName.get(InjectingObjectCodec.class),
            TypeName.get(encodedType.asType()),
            TypeName.get(dependency.asType())));
  }

  static MethodSpec.Builder initializeGetEncodedClassMethod(TypeElement encodedType) {
    return MethodSpec.methodBuilder("getEncodedClass")
        .addModifiers(Modifier.PUBLIC)
        .addAnnotation(Override.class)
        .returns(
            ParameterizedTypeName.get(
                ClassName.get(Class.class), TypeName.get(encodedType.asType())));
  }

  static MethodSpec.Builder initializeSerializeMethodBuilder(TypeElement encodedType) {
    return initializeSerializeMethodBuilder(encodedType, null);
  }

  /**
   * Initializes the appropriate deserialize method based on presence of dependency.
   *
   * <p>{@link InjectingObjectCodec#serialize} if dependency is non-null and {@link
   * ObjectCodec#serialize} otherwise.
   *
   * @param encodedType type being serialized
   * @param dependency type being injected
   */
  static MethodSpec.Builder initializeSerializeMethodBuilder(
      TypeElement encodedType, @Nullable TypeElement dependency) {
    MethodSpec.Builder builder =
        MethodSpec.methodBuilder("serialize")
            .addModifiers(Modifier.PUBLIC)
            .returns(void.class)
            .addAnnotation(Override.class)
            .addException(SerializationException.class)
            .addException(IOException.class);
    if (dependency != null) {
      builder.addParameter(TypeName.get(dependency.asType()), "dependency");
    }
    return builder
        .addParameter(TypeName.get(encodedType.asType()), "input")
        .addParameter(CodedOutputStream.class, "codedOut");
  }

  /** Initializes {@link ObjectCodec#deserialize}. */
  static MethodSpec.Builder initializeDeserializeMethodBuilder(TypeElement encodedType) {
    return initializeDeserializeMethodBuilder(encodedType, null);
  }

  /**
   * Initializes the appropriate deserialize method based on presence of dependency.
   *
   * <p>{@link InjectingObjectCodec#deserialize} if dependency is non-null and {@link
   * ObjectCodec#deserialize} otherwise.
   *
   * @param encodedType type being serialized
   * @param dependency type being injected
   */
  static MethodSpec.Builder initializeDeserializeMethodBuilder(
      TypeElement encodedType, @Nullable TypeElement dependency) {
    MethodSpec.Builder builder =
        MethodSpec.methodBuilder("deserialize")
            .addModifiers(Modifier.PUBLIC)
            .returns(TypeName.get(encodedType.asType()))
            .addAnnotation(Override.class)
            .addException(SerializationException.class)
            .addException(IOException.class);
    if (dependency != null) {
      builder.addParameter(TypeName.get(dependency.asType()), "dependency");
    }
    return builder.addParameter(CodedInputStream.class, "codedIn");
  }

  /**
   * Name of the generated codec class.
   *
   * <p>For {@code Foo.Bar} this is {@code Foo_Bar_AutoCodec}.
   */
  private static String getCodecName(Element element) {
    ImmutableList.Builder<String> classNamesBuilder = new ImmutableList.Builder<>();
    classNamesBuilder.add(GENERATED_CLASS_NAME_SUFFIX);
    do {
      classNamesBuilder.add(element.getSimpleName().toString());
      element = element.getEnclosingElement();
    } while (element instanceof TypeElement);
    return classNamesBuilder.build().reverse().stream().collect(Collectors.joining("_"));
  }
}
