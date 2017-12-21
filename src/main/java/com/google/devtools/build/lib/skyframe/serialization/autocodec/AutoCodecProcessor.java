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

import com.google.auto.service.AutoService;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.strings.StringCodecs;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.Processor;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.ElementFilter;
import javax.tools.Diagnostic;

/**
 * Javac annotation processor (compiler plugin) for generating {@link ObjectCodec} implementations.
 *
 * <p>User code must never reference this class.
 */
@AutoService(Processor.class)
public class AutoCodecProcessor extends AbstractProcessor {
  /**
   * Passing {@code --javacopt=-Aautocodec_print_generated} to {@code blaze build} tells AutoCodec
   * to print the generated code.
   */
  private static final String PRINT_GENERATED_OPTION = "autocodec_print_generated";

  private ProcessingEnvironment env; // Captured from `init` method.

  @Override
  public Set<String> getSupportedOptions() {
    return ImmutableSet.of(PRINT_GENERATED_OPTION);
  }

  @Override
  public Set<String> getSupportedAnnotationTypes() {
    return ImmutableSet.of(AutoCodecUtil.ANNOTATION.getCanonicalName());
  }

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported(); // Supports all versions of Java.
  }

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    this.env = processingEnv;
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    for (Element element : roundEnv.getElementsAnnotatedWith(AutoCodecUtil.ANNOTATION)) {
      AutoCodec annotation = element.getAnnotation(AutoCodecUtil.ANNOTATION);
      TypeElement encodedType = (TypeElement) element;
      TypeSpec.Builder codecClassBuilder = AutoCodecUtil.initializeCodecClassBuilder(encodedType);
      codecClassBuilder.addMethod(
          AutoCodecUtil.initializeGetEncodedClassMethod(encodedType)
              .addStatement("return $T.class", TypeName.get(encodedType.asType()))
              .build());
      switch (annotation.strategy()) {
        case CONSTRUCTOR:
          buildClassWithConstructorStrategy(codecClassBuilder, encodedType);
          break;
        case PUBLIC_FIELDS:
          buildClassWithPublicFieldsStrategy(codecClassBuilder, encodedType);
          break;
        case POLYMORPHIC:
          buildClassWithPolymorphicStrategy(codecClassBuilder, encodedType);
          break;
        default:
          throw new IllegalArgumentException("Unknown strategy: " + annotation.strategy());
      }
      String packageName =
          env.getElementUtils().getPackageOf(encodedType).getQualifiedName().toString();
      try {
        JavaFile file = JavaFile.builder(packageName, codecClassBuilder.build()).build();
        file.writeTo(env.getFiler());
        if (env.getOptions().containsKey(PRINT_GENERATED_OPTION)) {
          note("AutoCodec generated codec for " + encodedType + ":\n" + file);
        }
      } catch (IOException e) {
        env.getMessager()
            .printMessage(
                Diagnostic.Kind.ERROR, "Failed to generate output file: " + e.getMessage());
      }
    }
    return true;
  }

  private void buildClassWithConstructorStrategy(
      TypeSpec.Builder codecClassBuilder, TypeElement encodedType) {
    // In Java, every class has a constructor, so this always succeeds.
    ExecutableElement constructor =
        ElementFilter.constructorsIn(encodedType.getEnclosedElements()).get(0);
    List<? extends VariableElement> constructorParameters = constructor.getParameters();
    codecClassBuilder.addMethod(
        buildSerializeMethod(
            encodedType, constructorParameters, AutoCodecProcessor::paramNameAsGetter));
    MethodSpec.Builder deserializeBuilder =
        AutoCodecUtil.initializeDeserializeMethodBuilder(encodedType);
    buildDeserializeBody(deserializeBuilder, constructorParameters);
    addReturnNew(deserializeBuilder, encodedType, constructorParameters);
    codecClassBuilder.addMethod(deserializeBuilder.build());
  }

  private void buildClassWithPublicFieldsStrategy(
      TypeSpec.Builder codecClassBuilder, TypeElement encodedType) {
    List<? extends VariableElement> publicFields =
        ElementFilter.fieldsIn(env.getElementUtils().getAllMembers(encodedType))
            .stream()
            .filter(this::isPublicField)
            .collect(Collectors.toList());
    codecClassBuilder.addMethod(
        buildSerializeMethod(encodedType, publicFields, UnaryOperator.identity()));
    MethodSpec.Builder deserializeBuilder =
        AutoCodecUtil.initializeDeserializeMethodBuilder(encodedType);
    buildDeserializeBody(deserializeBuilder, publicFields);
    addInstantiatePopulateFieldsAndReturn(deserializeBuilder, encodedType, publicFields);
    codecClassBuilder.addMethod(deserializeBuilder.build());
  }

  /**
   * Heuristic that converts a constructor parameter to a getter.
   *
   * <p>For example, a parameter called {@code target} results in {@code getTarget()}.
   */
  private static String paramNameAsGetter(String name) {
    return "get" + name.substring(0, 1).toUpperCase() + name.substring(1) + "()";
  }

  private boolean isPublicField(VariableElement element) {
    if (matchesType(element.asType(), Void.class)) {
      return false; // Void types can't be instantiated, so the processor ignores them completely.
    }
    Set<Modifier> modifiers = element.getModifiers();
    return modifiers.contains(Modifier.PUBLIC) && !modifiers.contains(Modifier.STATIC);
  }

  private MethodSpec buildSerializeMethod(
      TypeElement encodedType,
      List<? extends VariableElement> parameters,
      UnaryOperator<String> nameToAccessor) {
    MethodSpec.Builder serializeBuilder =
        AutoCodecUtil.initializeSerializeMethodBuilder(encodedType);
    for (VariableElement parameter : parameters) {
      String paramAccessor = "input." + nameToAccessor.apply(parameter.getSimpleName().toString());
      TypeKind typeKind = parameter.asType().getKind();
      switch (typeKind) {
        case BOOLEAN:
          serializeBuilder.addStatement("codedOut.writeBoolNoTag($L)", paramAccessor);
          break;
        case INT:
          serializeBuilder.addStatement("codedOut.writeInt32NoTag($L)", paramAccessor);
          break;
        case DECLARED:
          buildSerializeBody(serializeBuilder, (DeclaredType) parameter.asType(), paramAccessor);
          break;
        default:
          throw new IllegalArgumentException("Unimplemented or invalid kind: " + typeKind);
      }
    }
    return serializeBuilder.build();
  }

  /**
   * Appends code statements to {@code builder} to serialize a pre-declared variable named {@code
   * accessor}.
   *
   * @param type the type of {@code accessor}
   * @param depth recursion depth of buildSerializeBody
   */
  private void buildSerializeBody(
      MethodSpec.Builder builder, DeclaredType type, String accessor, int depth) {
    builder.beginControlFlow("if ($L != null)", accessor); // Begin if not null block.
    builder.addStatement("codedOut.writeBoolNoTag(true)");
    // TODO(shahan): Add support for more types.
    if (isEnum(type)) {
      builder.addStatement("codedOut.writeInt32NoTag($L.ordinal())", accessor);
    } else if (matchesType(type, String.class)) {
      builder.addStatement(
          "$T.asciiOptimized().serialize($L, codedOut)", StringCodecs.class, accessor);
    } else if (matchesErased(type, Map.Entry.class)) {
      DeclaredType keyType = (DeclaredType) type.getTypeArguments().get(0);
      buildSerializeBody(builder, keyType, accessor + ".getKey()", depth + 1);
      DeclaredType valueType = (DeclaredType) type.getTypeArguments().get(1);
      buildSerializeBody(builder, valueType, accessor + ".getValue()", depth + 1);
    } else if (matchesErased(type, List.class) || matchesErased(type, ImmutableSortedSet.class)) {
      // Writes the target count to the stream so deserialization knows when to stop.
      builder.addStatement("codedOut.writeInt32NoTag($L.size())", accessor);
      DeclaredType repeatedType = (DeclaredType) type.getTypeArguments().get(0);
      String repeatedName = "repeated" + depth;
      builder.beginControlFlow(
          "for ($T $L : $L)", TypeName.get(repeatedType), repeatedName, accessor);
      buildSerializeBody(builder, repeatedType, repeatedName, depth + 1);
      builder.endControlFlow();
    } else {
      // Otherwise use the type's codec.
      builder.addStatement("$T.CODEC.serialize($L, codedOut)", TypeName.get(type), accessor);
    }
    builder.nextControlFlow("else");
    builder.addStatement("codedOut.writeBoolNoTag(false)");
    builder.endControlFlow(); // End if not null.
  }

  /** Convenience overload for depth = 0. */
  private void buildSerializeBody(MethodSpec.Builder builder, DeclaredType type, String accessor) {
    buildSerializeBody(builder, type, accessor, /*depth=*/ 0);
  }

  /**
   * Adds a body to the deserialize method that extracts serialized parameters.
   *
   * <p>Parameter values are extracted into local variables with the same name as the parameter
   * suffixed with a trailing underscore. For example, {@code target} becomes {@code target_}. This
   * is to avoid name collisions with variables used internally by AutoCodec.
   */
  private void buildDeserializeBody(
      MethodSpec.Builder builder, List<? extends VariableElement> parameters) {
    for (VariableElement parameter : parameters) {
      String paramName = parameter.getSimpleName() + "_";
      TypeKind typeKind = parameter.asType().getKind();
      switch (typeKind) {
        case BOOLEAN:
          builder.addStatement("boolean $L = codedIn.readBool()", paramName);
          break;
        case INT:
          builder.addStatement("int $L = codedIn.readInt32()", paramName);
          break;
        case DECLARED:
          buildDeserializeBody(builder, (DeclaredType) parameter.asType(), paramName);
          break;
        default:
          throw new IllegalArgumentException("Unimplemented or invalid kind: " + typeKind);
      }
    }
  }

  /**
   * Appends code statements to {@code builder}, declaring a variable called {@code name} and
   * initializing it with deserialization.
   *
   * @param type the type of {@code name}.
   * @param depth recursion depth of buildDeserializeBody.
   */
  private void buildDeserializeBody(
      MethodSpec.Builder builder, DeclaredType type, String name, int depth) {
    builder.addStatement("$T $L = null", TypeName.get(type), name);
    builder.beginControlFlow("if (codedIn.readBool())"); // Begin null-handling block.
    // TODO(shahan): Add support for more types.
    if (isEnum(type)) {
      // TODO(shahan): memoize this expensive call to values().
      builder.addStatement("$L = $T.values()[codedIn.readInt32()]", name, TypeName.get(type));
    } else if (matchesType(type, String.class)) {
      builder.addStatement(
          "$L = $T.asciiOptimized().deserialize(codedIn)", name, StringCodecs.class);
    } else if (matchesErased(type, Map.Entry.class)) {
      DeclaredType keyType = (DeclaredType) type.getTypeArguments().get(0);
      String keyName = "key" + depth;
      buildDeserializeBody(builder, keyType, keyName, depth + 1);
      DeclaredType valueType = (DeclaredType) type.getTypeArguments().get(1);
      String valueName = "value" + depth;
      buildDeserializeBody(builder, valueType, valueName, depth + 1);
      builder.addStatement("$L = $T.immutableEntry($L, $L)", name, Maps.class, keyName, valueName);
    } else if (matchesErased(type, List.class)) {
      builder.addStatement("$L = new $T<>()", name, ArrayList.class);
      String lengthName = "length" + depth;
      builder.addStatement("int $L = codedIn.readInt32()", lengthName);
      String indexName = "i" + depth;
      builder.beginControlFlow(
          "for (int $L = 0; $L < $L; ++$L)", indexName, indexName, lengthName, indexName);
      DeclaredType repeatedType = (DeclaredType) type.getTypeArguments().get(0);
      String repeatedName = "repeated" + depth;
      buildDeserializeBody(builder, repeatedType, repeatedName, depth + 1);
      builder.addStatement("$L.add($L)", name, repeatedName);
      builder.endControlFlow();
    } else if (matchesErased(type, ImmutableSortedSet.class)) {
      DeclaredType repeatedType = (DeclaredType) type.getTypeArguments().get(0);
      builder.addStatement(
          "$T<$T> builder = new $T<>($T.naturalOrder())",
          ImmutableSortedSet.Builder.class,
          TypeName.get(repeatedType),
          ImmutableSortedSet.Builder.class,
          Comparator.class);
      String lengthName = "length" + depth;
      builder.addStatement("int $L = codedIn.readInt32()", lengthName);
      String indexName = "i" + depth;
      builder.beginControlFlow(
          "for (int $L = 0; $L < $L; ++$L)", indexName, indexName, lengthName, indexName);
      String repeatedName = "repeated" + depth;
      buildDeserializeBody(builder, repeatedType, repeatedName, depth + 1);
      builder.addStatement("builder.add($L)", repeatedName);
      builder.endControlFlow();
      builder.addStatement("$L = builder.build()", name);
    } else {
      // Otherwise, use the type's codec.
      builder.addStatement("$L = $T.CODEC.deserialize(codedIn)", name, TypeName.get(type));
    }
    builder.endControlFlow(); // End null-handling block.
  }

  /** Overload of above, for common case of 0 depth. */
  private void buildDeserializeBody(MethodSpec.Builder builder, DeclaredType type, String name) {
    buildDeserializeBody(builder, type, name, /*depth=*/ 0);
  }

  /**
   * Invokes the constructor and returns the value.
   *
   * <p>Used by the {@link AutoCodec.Strategy.CONSTRUCTOR} strategy.
   */
  private static void addReturnNew(
      MethodSpec.Builder builder, TypeElement type, List<? extends VariableElement> parameters) {
    builder.addStatement(
        "return new $T($L)",
        TypeName.get(type.asType()),
        parameters.stream().map(p -> p.getSimpleName() + "_").collect(Collectors.joining(", ")));
  }

  /**
   * Invokes the constructor, populates public fields and returns the value.
   *
   * <p>Used by the {@link AutoCodec.Strategy.PUBLIC_FIELDS} strategy.
   */
  private static void addInstantiatePopulateFieldsAndReturn(
      MethodSpec.Builder builder, TypeElement type, List<? extends VariableElement> fields) {
    builder.addStatement(
        "$T deserializationResult = new $T()",
        TypeName.get(type.asType()),
        TypeName.get(type.asType()));
    for (VariableElement field : fields) {
      String fieldName = field.getSimpleName().toString();
      builder.addStatement("deserializationResult.$L = $L", fieldName, fieldName + "_");
    }
    builder.addStatement("return deserializationResult");
  }

  private static void buildClassWithPolymorphicStrategy(
      TypeSpec.Builder codecClassBuilder, TypeElement encodedType) {
    if (!encodedType.getModifiers().contains(Modifier.ABSTRACT)) {
      throw new IllegalArgumentException(
          encodedType + " is not abstract, but POLYMORPHIC was selected as the strategy.");
    }
    codecClassBuilder.addMethod(buildPolymorphicSerializeMethod(encodedType));
    codecClassBuilder.addMethod(buildPolymorphicDeserializeMethod(encodedType));
  }

  private static MethodSpec buildPolymorphicSerializeMethod(TypeElement encodedType) {
    MethodSpec.Builder builder = AutoCodecUtil.initializeSerializeMethodBuilder(encodedType);
    builder.beginControlFlow("if (input != null)");
    builder.addStatement("Class<?> clazz = input.getClass()");
    builder.beginControlFlow("try");
    builder.addStatement("$T codecField = clazz.getDeclaredField(\"CODEC\")", Field.class);
    builder.addStatement("codedOut.writeBoolNoTag(true)");
    builder.addStatement(
        "$T.asciiOptimized().serialize(clazz.getName(), codedOut)", StringCodecs.class);
    builder.addStatement("Object codec = codecField.get(null)");
    builder.addStatement(
        "$T serializeMethod = codec.getClass().getDeclaredMethod(\"serialize\", clazz, $T.class)",
        Method.class,
        CodedOutputStream.class);
    builder.addStatement("serializeMethod.invoke(codec, input, codedOut)");
    builder.nextControlFlow(
        "catch ($T|$T|$T|$T e)",
        NoSuchFieldException.class,
        NoSuchMethodException.class,
        IllegalAccessException.class,
        InvocationTargetException.class);
    builder.addStatement("throw new SerializationException(input.getClass().getName(), e)");
    builder.endControlFlow();
    builder.nextControlFlow("else");
    builder.addStatement("codedOut.writeBoolNoTag(false)");
    builder.endControlFlow();
    return builder.build();
  }

  private static MethodSpec buildPolymorphicDeserializeMethod(TypeElement encodedType) {
    MethodSpec.Builder builder = AutoCodecUtil.initializeDeserializeMethodBuilder(encodedType);
    builder.addStatement("$T deserialized = null", TypeName.get(encodedType.asType()));
    builder.beginControlFlow("if (codedIn.readBool())");
    builder.addStatement(
        "String className = $T.asciiOptimized().deserialize(codedIn)", StringCodecs.class);
    builder.beginControlFlow("try");
    builder.addStatement("Class<?> clazz = Class.forName(className)", StringCodecs.class);
    builder.addStatement("Object codec = clazz.getDeclaredField(\"CODEC\").get(null)");
    builder.addStatement(
        "$T deserializeMethod = codec.getClass().getDeclaredMethod(\"deserialize\", $T.class)",
        Method.class,
        CodedInputStream.class);
    builder.addStatement(
        "deserialized = ($T)deserializeMethod.invoke(codec, codedIn)",
        TypeName.get(encodedType.asType()));
    builder.nextControlFlow(
        "catch ($T|$T|$T|$T|$T e)",
        ClassNotFoundException.class,
        NoSuchFieldException.class,
        NoSuchMethodException.class,
        IllegalAccessException.class,
        InvocationTargetException.class);
    builder.addStatement("throw new SerializationException(className, e)");
    builder.endControlFlow();
    builder.endControlFlow();
    builder.addStatement("return deserialized");
    return builder.build();
  }

  /** True when {@code type} has the same type as {@code clazz}. */
  private boolean matchesType(TypeMirror type, Class<?> clazz) {
    return env.getTypeUtils()
        .isSameType(
            type, env.getElementUtils().getTypeElement((clazz.getCanonicalName())).asType());
  }

  /** True when erasure of {@code type} matches erasure of {@code clazz}. */
  private boolean matchesErased(TypeMirror type, Class<?> clazz) {
    return env.getTypeUtils()
        .isSameType(
            env.getTypeUtils().erasure(type),
            env.getTypeUtils()
                .erasure(
                    env.getElementUtils().getTypeElement((clazz.getCanonicalName())).asType()));
  }

  private boolean isEnum(TypeMirror type) {
    return env.getTypeUtils().asElement(type).getKind() == ElementKind.ENUM;
  }

  /** Emits a note to BUILD log during annotation processing for debugging. */
  private void note(String note) {
    env.getMessager().printMessage(Diagnostic.Kind.NOTE, note);
  }
}
