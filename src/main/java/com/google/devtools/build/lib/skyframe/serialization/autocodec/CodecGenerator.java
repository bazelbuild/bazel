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

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.squareup.javapoet.MethodSpec;
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
    List<? extends FieldGenerator> fieldGenerators = getFieldGenerators(encodedType);

    TypeSpec.Builder classBuilder = initializeCodecClassBuilder(encodedType, env);
    performAdditionalCodecInitialization(classBuilder, encodedType, instantiator);

    MethodSpec.Builder constructor = initializeConstructor(encodedType, fieldGenerators.size());
    MethodSpec.Builder serialize = initializeSerializeMethodBuilder(encodedType, annotation, env);
    MethodSpec.Builder deserialize = initializeDeserializeMethod(encodedType);

    for (FieldGenerator generator : fieldGenerators) {
      generator.generateHandleMember(classBuilder, constructor);
      generator.generateAdditionalMemberVariables(classBuilder);
      generator.generateConstructorCode(constructor);
      generator.generateSerializeCode(serialize);
      generator.generateDeserializeCode(deserialize);
    }

    addImplementationToEndOfMethods(
        instantiator, constructor, deserialize, !fieldGenerators.isEmpty());

    return classBuilder
        .addMethod(constructor.build())
        .addMethod(serialize.build())
        .addMethod(deserialize.build())
        .build();
  }

  /**
   * Performs additional initialization steps on the codec being created.
   *
   * <p>Adds the correct superclass. May define additional field-independent methods.
   */
  abstract void performAdditionalCodecInitialization(
      TypeSpec.Builder classBuilder, TypeElement encodedType, ExecutableElement instantiator);

  /** Creates {@link FieldGenerator} instances that generate code for serialized fields. */
  abstract List<? extends FieldGenerator> getFieldGenerators(TypeElement type)
      throws SerializationProcessingException;

  /**
   * Initializes the field-independent parts of the constructor.
   *
   * @param fieldCount number of fields to serialize. This is used in two ways. 1. Exception
   *     handling logic may depend on the presence of fields. 2. We cross check the number of fields
   *     at runtime.
   */
  abstract MethodSpec.Builder initializeConstructor(TypeElement type, int fieldCount);

  /** Initializes the method that performs deserialization work. */
  abstract MethodSpec.Builder initializeDeserializeMethod(TypeElement encodedType);

  /**
   * Adds field-independent code at the end of methods after per-field code is added.
   *
   * @param hasFields true if there are any fields to serialize, based on the result of {@link
   *     #getFieldGenerators}. Exception handling logic may depend on the presence of fields.
   */
  abstract void addImplementationToEndOfMethods(
      ExecutableElement instantiator,
      MethodSpec.Builder constructor,
      MethodSpec.Builder deserialize,
      boolean hasFields);
}
