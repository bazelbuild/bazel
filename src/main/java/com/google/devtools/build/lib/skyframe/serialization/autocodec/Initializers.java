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

import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getErasure;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getGeneratedName;

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedOutputStream;
import com.squareup.javapoet.AnnotationSpec;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import javax.annotation.processing.Generated;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.TypeMirror;

/** Methods that initialize generated type and method builders. */
class Initializers {
  // Synthesized classes will have `_AutoCodec` suffix added.
  private static final String GENERATED_CLASS_NAME_SUFFIX = "AutoCodec";

  /**
   * Initializes a builder for a class of the appropriate type.
   *
   * @param encodedType type being serialized
   */
  static TypeSpec.Builder initializeCodecClassBuilder(
      TypeElement encodedType, AutoCodecAnnotation annotation, ProcessingEnvironment env) {
    return TypeSpec.classBuilder(getGeneratedName(encodedType, GENERATED_CLASS_NAME_SUFFIX))
        .addAnnotation(
            AnnotationSpec.builder(Generated.class)
                .addMember("value", "$S", AutoCodecProcessor.class.getCanonicalName())
                .build())
        .addAnnotation(
            AnnotationSpec.builder(SuppressWarnings.class)
                .addMember("value", "$S", "unchecked")
                .addMember("value", "$S", "rawtypes")
                .addMember("value", "$S", "removal")
                .build())
        .addMethod(defineGetEncodedClassMethod(encodedType, env))
        .addMethod(defineAutoRegisterMethod(annotation));
  }

  /** Initializes the {@link ObjectCodec#serialize} method. */
  static MethodSpec.Builder initializeSerializeMethodBuilder(
      TypeElement encodedType, AutoCodecAnnotation annotation, ProcessingEnvironment env) {
    MethodSpec.Builder builder =
        MethodSpec.methodBuilder("serialize")
            .addModifiers(Modifier.PUBLIC)
            .returns(void.class)
            .addAnnotation(Override.class)
            .addAnnotation(
                AnnotationSpec.builder(ClassName.get(SuppressWarnings.class))
                    .addMember("value", "$S", "unchecked")
                    .build())
            .addException(SerializationException.class)
            .addException(IOException.class)
            .addParameter(SerializationContext.class, "context")
            .addParameter(getErasure(encodedType, env), "obj")
            .addParameter(CodedOutputStream.class, "codedOut");
    if (annotation.checkClassExplicitlyAllowed()) {
      builder.addStatement("context.checkClassExplicitlyAllowed(getEncodedClass(), obj)");
    }
    for (TypeMirror explicitlyAllowedClass : annotation.explicitlyAllowClass()) {
      builder.addStatement("context.addExplicitlyAllowedClass($T.class)", explicitlyAllowedClass);
    }
    return builder;
  }

  /** Defines the link {@link ObjectCodec#getEncodedClass} method. */
  private static MethodSpec defineGetEncodedClassMethod(
      TypeElement encodedType, ProcessingEnvironment env) {
    TypeName returnType = getErasure(encodedType, env);
    return MethodSpec.methodBuilder("getEncodedClass")
        .addModifiers(Modifier.PUBLIC)
        .addAnnotation(Override.class)
        .returns(ParameterizedTypeName.get(ClassName.get(Class.class), returnType))
        .addStatement("return $T.class", returnType)
        .build();
  }

  private static MethodSpec defineAutoRegisterMethod(AutoCodecAnnotation annotation) {
    return MethodSpec.methodBuilder("autoRegister")
        .addModifiers(Modifier.PUBLIC)
        .addAnnotation(Override.class)
        .returns(TypeName.BOOLEAN)
        .addStatement("return $L", annotation.autoRegister())
        .build();
  }

  private Initializers() {}
}
