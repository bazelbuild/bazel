// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.skyframe.serialization.autocodec.Initializers.initializeCodecClassBuilder;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.Initializers.initializeSerializeMethodBuilder;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getErasure;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.util.List;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.TypeElement;

/** Defines an abstract strategy for generating {@link ObjectCodec} implementations. */
// TODO(b/297857068): migrate other types to this class.
abstract class CodecGenerator {
  final ProcessingEnvironment env;

  CodecGenerator(ProcessingEnvironment env) {
    this.env = env;
  }

  /** Creates the codec by delegating lower level implementation methods. */
  final TypeSpec defineCodec(
      TypeElement encodedType, AutoCodec annotation, ExecutableElement instantiator)
      throws SerializationProcessingException {
    ImmutableList<FieldGenerator> fieldGenerators = getFieldGenerators(encodedType);

    TypeSpec.Builder classBuilder = initializeCodecClassBuilder(encodedType, env);
    TypeName encodedTypeName = getErasure(encodedType, env);
    performAdditionalCodecInitialization(
        classBuilder, encodedTypeName, instantiator, fieldGenerators);

    MethodSpec.Builder constructor = initializeConstructor(encodedType, fieldGenerators);
    MethodSpec.Builder serialize = initializeSerializeMethodBuilder(encodedType, annotation, env);
    MethodSpec.Builder deserialize = initializeDeserializeMethod(encodedTypeName);

    for (FieldGenerator generator : fieldGenerators) {
      generator.generateHandleMember(classBuilder, constructor);
      generator.generateAdditionalMemberVariables(classBuilder);
      generator.generateConstructorCode(constructor);
      generator.generateSerializeCode(serialize);
      generator.generateDeserializeCode(deserialize);
    }

    addImplementationToEndOfMethods(constructor, deserialize, fieldGenerators);

    return classBuilder
        .addMethod(constructor.build())
        .addMethod(serialize.build())
        .addMethod(deserialize.build())
        .build();
  }

  /** Creates {@link FieldGenerator} instances that generate code for serialized fields. */
  abstract ImmutableList<FieldGenerator> getFieldGenerators(TypeElement type)
      throws SerializationProcessingException;

  /**
   * Performs additional initialization steps on the codec being created.
   *
   * <p>Adds the correct superclass. May define additional field-independent methods.
   */
  abstract void performAdditionalCodecInitialization(
      TypeSpec.Builder classBuilder,
      TypeName encodedTypeName,
      ExecutableElement instantiator,
      List<? extends FieldGenerator> fieldGenerators);

  abstract void generateConstructorPreamble(
      TypeElement encodedType,
      ImmutableList<FieldGenerator> fieldGenerators,
      MethodSpec.Builder constructor);

  /** Initializes the method that performs deserialization work. */
  abstract MethodSpec.Builder initializeDeserializeMethod(TypeName typeName);

  /** Adds field-independent code at the end of methods after per-field code is added. */
  abstract void addImplementationToEndOfMethods(
      MethodSpec.Builder constructor,
      MethodSpec.Builder deserialize,
      ImmutableList<FieldGenerator> fieldGenerators);

  /** Initializes the (mostly) field-independent parts of the constructor. */
  private final MethodSpec.Builder initializeConstructor(
      TypeElement encodedType, ImmutableList<FieldGenerator> fieldGenerators) {
    MethodSpec.Builder constructor = MethodSpec.constructorBuilder();
    generateConstructorPreamble(encodedType, fieldGenerators, constructor);

    if (fieldGenerators.stream().anyMatch(g -> g.getGetterName() == null)) {
      // If there are any fields not retrieved by getters, the per-field section of the constructor
      // will perform reflective operations to obtain handles to the variables. These are enclosed
      // in a common try-catch block.
      constructor.beginControlFlow("try");
    }
    return constructor;
  }
}
