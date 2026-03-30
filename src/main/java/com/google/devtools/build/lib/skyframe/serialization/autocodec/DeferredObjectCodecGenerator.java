// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.BUILDER_NAME;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.BUILDER_TYPE_NAME;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.CONSTRUCTOR_LOOKUP_NAME;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.makeGetterName;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.DeferredObjectCodecConstants.makeSetterName;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.findRelationWithGenerics;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getClassLineage;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getErasure;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.Relation;
import com.google.protobuf.CodedInputStream;
import com.squareup.javapoet.AnnotationSpec;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.FieldSpec;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import com.squareup.javapoet.TypeVariableName;
import com.squareup.javapoet.WildcardTypeName;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.Name;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.ElementFilter;
import javax.lang.model.util.Elements;

/** Generates general purpose {@link AutoCodec} codecs using {@link DeferredObjectCodec}. */
final class DeferredObjectCodecGenerator extends CodecGenerator {
  /**
   * If {@link AutoCodec#deserializedInterface()} is non-empty, this is the name of the class
   * generated for deserialized objects. Note that this is a static nested class inside the
   * generated codec class.
   */
  private static final String DESERIALIZED_CLASS_NAME = "Deserialized";

  /** Parameters of the constructor annotated by the codec, used to define the serialized fields. */
  private final List<? extends VariableElement> parameters;

  DeferredObjectCodecGenerator(
      ProcessingEnvironment env, List<? extends VariableElement> parameters) {
    super(env);
    this.parameters = checkNotNull(parameters);
  }

  @Override
  ImmutableList<FieldGenerator> getFieldGenerators(TypeElement encodedType)
      throws SerializationProcessingException {
    var generators = ImmutableList.<FieldGenerator>builder();
    for (VariableElement parameter : parameters) {
      FieldGenerator generator = getFieldGenerator(encodedType, parameter);
      if (generator == null) {
        throw new SerializationProcessingException(
            encodedType,
            "%s: No getter found corresponding to parameter %s, %s",
            encodedType,
            parameter.getSimpleName(),
            parameter.asType());
      }
      generators.add(generator);
    }
    return generators.build();
  }

  @Override
  void performAdditionalCodecInitialization(
      TypeSpec.Builder classBuilder,
      TypeElement encodedType,
      ExecutableElement instantiator,
      AutoCodecAnnotation annotation,
      List<? extends FieldGenerator> fieldGenerators) {
    TypeName encodedTypeName = getErasure(encodedType, env);
    classBuilder
        .superclass(
            ParameterizedTypeName.get(ClassName.get(DeferredObjectCodec.class), encodedTypeName))
        .addType(defineBuilder(encodedTypeName, annotation, fieldGenerators, instantiator));
    if (annotation.deserializedInterface().isPresent()) {
      classBuilder
          .addType(
              defineDeserializedClass(
                  encodedType, annotation.deserializedInterface().get(), instantiator))
          .addMethod(defineAdditionalEncodedClassesMethod(encodedTypeName));
    }
  }

  @Override
  void generateConstructorPreamble(
      TypeElement encodedType,
      ImmutableList<FieldGenerator> fieldGenerators,
      MethodSpec.Builder constructor) {
    // Adds constructor-scoped MethodHandles.Lookup variables named `CONSTRUCTOR_LOOKUP_NAME<N>`.
    // <N> is the hierarchy level of the lookup.
    ImmutableList<TypeElement> lineage = getClassLineage(encodedType, env);
    for (int i = 0; i < lineage.size(); i++) {
      constructor.addStatement("$T $L", MethodHandles.Lookup.class, CONSTRUCTOR_LOOKUP_NAME + i);
    }
    constructor.beginControlFlow("try");
    for (int i = 0; i < lineage.size(); i++) {
      constructor.addStatement(
          "$L = $T.privateLookupIn($T.class, $T.lookup())",
          CONSTRUCTOR_LOOKUP_NAME + i,
          MethodHandles.class,
          getErasure(lineage.get(i), env),
          MethodHandles.class);
    }
    constructor
        .nextControlFlow("catch ($T e)", IllegalAccessException.class)
        .addStatement("throw new $T(e)", IllegalStateException.class)
        .endControlFlow();
  }

  /** Initializes the {@link DeferredObjectCodec#deserializeDeferred} method. */
  @Override
  MethodSpec.Builder initializeDeserializeMethod(TypeName typeName) {
    return MethodSpec.methodBuilder("deserializeDeferred")
        .addModifiers(Modifier.PUBLIC)
        .returns(DeferredObjectCodec.DeferredValue.class)
        .addAnnotation(Override.class)
        .addException(SerializationException.class)
        .addException(IOException.class)
        .addParameter(AsyncDeserializationContext.class, "context")
        .addParameter(CodedInputStream.class, "codedIn")
        .addStatement("var $L = new $L()", BUILDER_NAME, BUILDER_TYPE_NAME);
  }

  @Override
  void addImplementationToEndOfMethods(
      MethodSpec.Builder constructor,
      MethodSpec.Builder deserialize,
      ImmutableList<FieldGenerator> fieldGenerators) {
    if (fieldGenerators.stream().anyMatch(g -> g.getGetterName() == null)) {
      // There are fields not retrieved by getters. The per-field section of the constructor
      // performs reflective operations to obtain handles to the variables. These are enclosed in a
      // common try-catch block.
      constructor
          .nextControlFlow(
              "catch ($T|$T e)", NoSuchFieldException.class, IllegalAccessException.class)
          .addStatement("throw new $T(e)", AssertionError.class)
          .endControlFlow();
    }
    deserialize.addStatement("return $L", BUILDER_NAME);
  }

  /** Defines a suitable {@link DeferredObjectCodec.DeferredValue} instance. */
  private TypeSpec defineBuilder(
      TypeName encodedTypeName,
      AutoCodecAnnotation annotation,
      List<? extends FieldGenerator> fieldGenerators,
      ExecutableElement instantiator) {
    TypeSpec.Builder classBuilder =
        TypeSpec.classBuilder(BUILDER_TYPE_NAME)
            .addAnnotation(
                AnnotationSpec.builder(ClassName.get(SuppressWarnings.class))
                    .addMember("value", "$S", "unchecked")
                    .addMember("value", "$S", "rawtypes")
                    .build())
            .addSuperinterface(
                ParameterizedTypeName.get(
                    ClassName.get(DeferredObjectCodec.DeferredValue.class), encodedTypeName));
    for (FieldGenerator field : fieldGenerators) {
      // There's no risk of shadowing here because all parameters are arguments of the instantiator.
      // So the field parameter name can be used directly as a field name in the builder.

      // Defines a member variable for each variable.
      classBuilder.addField(
          FieldSpec.builder(
                  field.getTypeName(), field.getParameterName().toString(), Modifier.PRIVATE)
              .build());

      TypeKind kind = field.getType().getKind();
      if (kind.isPrimitive() || kind.equals(TypeKind.ARRAY)) {
        // Skips adding setters for primitives or arrays. Primitives are read immediately from
        // the stream and partially initialized arrays are immediately available. Deserialization
        // assigns these values directly, without the need for a setter.
        continue;
      }

      // Adds a static setter for each variable.
      classBuilder.addMethod(
          MethodSpec.methodBuilder(makeSetterName(field.getParameterName()))
              .addModifiers(Modifier.PRIVATE, Modifier.STATIC)
              .returns(void.class)
              .addParameter(ClassName.get(/* packageName= */ "", BUILDER_TYPE_NAME), "builder")
              .addParameter(TypeName.get(Object.class), "value")
              .addStatement(
                  "builder.$L = ($T) value", field.getParameterName(), field.getTypeName())
              .build());
    }

    // Defines the call method that invokes the instantiator.
    MethodSpec.Builder callMethod =
        MethodSpec.methodBuilder("call")
            .addModifiers(Modifier.PUBLIC)
            .returns(encodedTypeName)
            .addAnnotation(Override.class);

    String parameters =
        fieldGenerators.stream()
            .map(generator -> generator.getParameterName().toString())
            .collect(joining(", "));
    if (annotation.deserializedInterface().isPresent()) {
      callMethod.addStatement("return new $L($L)", DESERIALIZED_CLASS_NAME, parameters);
    } else if (instantiator.getKind().equals(ElementKind.CONSTRUCTOR)) {
      callMethod.addStatement("return new $T($L)", encodedTypeName, parameters);
    } else { // a factory method otherwise
      callMethod.addStatement(
          "return $T.$L($L)", encodedTypeName, instantiator.getSimpleName(), parameters);
    }

    classBuilder.addMethod(callMethod.build());

    return classBuilder.build();
  }

  /**
   * Determines how the code for {@code parameter} will be serialized.
   *
   * <p>First checks for a matching field, falling back to finding a matching getter if a field
   * cannot be found. Returns null if both tactics fail to match.
   */
  @Nullable
  private FieldGenerator getFieldGenerator(TypeElement encodedType, VariableElement parameter)
      throws SerializationProcessingException {
    Elements elements = env.getElementUtils();

    ImmutableList<TypeElement> lineage = getClassLineage(encodedType, env);
    for (int i = 0; i < lineage.size(); i++) {
      TypeElement type = lineage.get(i);
      Optional<VariableElement> field =
          ElementFilter.fieldsIn(type.getEnclosedElements()).stream()
              .filter(f -> f.getSimpleName().equals(parameter.getSimpleName()))
              .findAny();
      if (field.isEmpty()) {
        continue;
      }
      Relation relation = findRelationWithGenerics(field.get().asType(), parameter.asType(), env);
      if (relation == Relation.UNRELATED_TO) {
        throw new SerializationProcessingException(
            encodedType,
            "%s: parameter %s's type %s is unrelated to corresponding field type %s",
            encodedType.getQualifiedName(),
            parameter.getSimpleName(),
            parameter.asType(),
            field.get().asType());
      }
      return DeferredObjectCodecFieldGenerators.create(
          parameter.getSimpleName(),
          parameter.asType(),
          ClassName.get(type),
          i,
          new DeferredObjectCodecFieldGenerators.FieldType(getErasure(field.get().asType(), env)),
          env);
    }

    // No matching member variable was found. Falls back on finding a matching getter instead. This
    // is the case for AutoValue fields.
    String expectedGetterName = makeGetterName(parameter); // e.g. getFoo(), isFoo()
    String propertyAccessorName = parameter.getSimpleName().toString(); // e.g. foo()

    for (int i = 0; i < lineage.size(); i++) {
      TypeElement type = lineage.get(i);
      for (ExecutableElement method : ElementFilter.methodsIn(elements.getAllMembers(type))) {
        if (isMethodMatchingGetter(
            expectedGetterName, propertyAccessorName, parameter.asType(), method)) {
          return DeferredObjectCodecFieldGenerators.create(
              parameter.getSimpleName(),
              parameter.asType(),
              ClassName.get(type),
              i,
              new DeferredObjectCodecFieldGenerators.GetterName(method.getSimpleName().toString()),
              env);
        }
      }
    }
    return null;
  }

  /** Checks if {@code method} is a suitable getter. */
  private boolean isMethodMatchingGetter(
      String expectedGetterName,
      String propertyAccessorName,
      TypeMirror parameterType,
      ExecutableElement method) {
    Name methodName = method.getSimpleName();
    if (!(methodName.contentEquals(expectedGetterName)
        || methodName.contentEquals(propertyAccessorName))) {
      return false;
    }
    if (!method.getParameters().isEmpty()) {
      return false;
    }
    return findRelationWithGenerics(parameterType, method.getReturnType(), env)
        != Relation.UNRELATED_TO;
  }

  private static TypeSpec defineDeserializedClass(
      TypeElement encodedType, TypeMirror deserializedInterface, ExecutableElement instantiator) {
    MethodSpec constructor =
        MethodSpec.constructorBuilder()
            .addParameters(instantiator.getParameters().stream().map(ParameterSpec::get).toList())
            .addStatement(
                instantiator.getParameters().stream()
                    .map(VariableElement::getSimpleName)
                    .collect(joining(", ", "super(", ")")))
            .build();
    return TypeSpec.classBuilder(DESERIALIZED_CLASS_NAME)
        .addModifiers(Modifier.STATIC)
        .addTypeVariables(
            encodedType.getTypeParameters().stream().map(TypeVariableName::get).toList())
        .superclass(TypeName.get(encodedType.asType()))
        .addSuperinterface(TypeName.get(deserializedInterface))
        .addMethod(constructor)
        .build();
  }

  private MethodSpec defineAdditionalEncodedClassesMethod(TypeName encodedTypeName) {
    return MethodSpec.methodBuilder("additionalEncodedClasses")
        .addModifiers(Modifier.PUBLIC)
        .addAnnotation(Override.class)
        .returns(
            // The following is a really long way to write `ImmutableSet<Class<? extends T>>`
            // where T is `encodedTypeName`.
            ParameterizedTypeName.get(
                ClassName.get(ImmutableSet.class),
                ParameterizedTypeName.get(
                    ClassName.get(Class.class), WildcardTypeName.subtypeOf(encodedTypeName))))
        .addStatement("return $T.of($L.class)", ImmutableSet.class, DESERIALIZED_CLASS_NAME)
        .build();
  }
}
