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

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getClassLineage;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getErasure;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.isSerializableField;
import static javax.lang.model.util.ElementFilter.fieldsIn;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.InterningObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.unsafe.UnsafeProvider;
import com.google.protobuf.CodedInputStream;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.util.List;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;

/** Generates instances of {@link InterningObjectCodec}. */
final class InterningObjectCodecGenerator extends CodecGenerator {

  InterningObjectCodecGenerator(ProcessingEnvironment env) {
    super(env);
  }

  @Override
  ImmutableList<FieldGenerator> getFieldGenerators(TypeElement type)
      throws SerializationProcessingException {
    ImmutableList<TypeElement> types = getClassLineage(type, env);

    ImmutableList.Builder<FieldGenerator> result = ImmutableList.builder();
    // Iterates in reverse order so variables are ordered highest superclass first, as they would
    // be ordered in the class layout.
    for (int i = types.size() - 1; i >= 0; i--) {
      for (VariableElement variable : fieldsIn(types.get(i).getEnclosedElements())) {
        if (!isSerializableField(variable)) {
          continue;
        }
        result.add(InterningObjectCodecFieldGenerators.create(variable, i, env));
      }
    }
    return result.build();
  }

  @Override
  void performAdditionalCodecInitialization(
      TypeSpec.Builder classBuilder,
      TypeName encodedTypeName,
      ExecutableElement internMethod,
      List<? extends FieldGenerator> unusedFieldGenerators) {
    classBuilder.superclass(
        ParameterizedTypeName.get(ClassName.get(InterningObjectCodec.class), encodedTypeName));

    // Defines the `InterningObjectCodec.intern` implementation.
    VariableElement param = getOnlyElement(internMethod.getParameters());
    classBuilder.addMethod(
        MethodSpec.methodBuilder("intern")
            .addModifiers(Modifier.PUBLIC)
            .addAnnotation(Override.class)
            .addParameter(getErasure(param.asType(), env), "value")
            .returns(encodedTypeName)
            .addStatement("return $T.$L(value)", encodedTypeName, internMethod.getSimpleName())
            .build());
  }

  @Override
  void generateConstructorPreamble(
      TypeElement encodedType,
      ImmutableList<FieldGenerator> fieldGenerators,
      MethodSpec.Builder constructor) {
    TypeName encodedTypeName = getErasure(encodedType, env);
    constructor.addStatement(
        "int runtimeFieldCount = $T.getSerializableFieldCount($T.class)",
        RuntimeHelpers.class,
        encodedTypeName);
    constructor
        .beginControlFlow("if (runtimeFieldCount != $L)", fieldGenerators.size())
        .addStatement(
            "throw new IllegalStateException(\"$T's AutoCodec expected $L fields, but there were"
                + " \" + runtimeFieldCount + \" serializable fields at runtime. See"
                + " b/319301818 for explanation and workaround.\")",
            encodedTypeName,
            fieldGenerators.size())
        .endControlFlow();
  }

  /** Initializes the {@link InterningObjectCodec#deserializeInterned} method. */
  @Override
  MethodSpec.Builder initializeDeserializeMethod(TypeName typeName) {
    return MethodSpec.methodBuilder("deserializeInterned")
        .addModifiers(Modifier.PUBLIC)
        .returns(typeName)
        .addAnnotation(Override.class)
        .addException(SerializationException.class)
        .addException(IOException.class)
        .addParameter(AsyncDeserializationContext.class, "context")
        .addParameter(CodedInputStream.class, "codedIn")
        .addStatement("$T instance", typeName)
        .beginControlFlow("try")
        .addStatement(
            "instance = ($T) $T.unsafe().allocateInstance($T.class)",
            typeName,
            UnsafeProvider.class,
            typeName)
        .nextControlFlow("catch ($T e)", ReflectiveOperationException.class)
        .addStatement("throw new $T(e)", IllegalStateException.class)
        .endControlFlow();
  }

  @Override
  void addImplementationToEndOfMethods(
      MethodSpec.Builder constructor,
      MethodSpec.Builder deserialize,
      ImmutableList<FieldGenerator> fieldGenerators) {
    if (!fieldGenerators.isEmpty()) {
      constructor
          .nextControlFlow("catch ($T e)", NoSuchFieldException.class)
          .addStatement("throw new $T(e)", AssertionError.class)
          .endControlFlow();
    }
    deserialize.addStatement("return instance");
  }
}
