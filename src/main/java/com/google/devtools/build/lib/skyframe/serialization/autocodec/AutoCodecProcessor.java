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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.service.AutoService;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
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
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.Processor;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
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
  private Marshallers marshallers;

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
    this.marshallers = new Marshallers(processingEnv);
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
    List<ExecutableElement> constructors =
        ElementFilter.constructorsIn(encodedType.getEnclosedElements());
    ImmutableList<ExecutableElement> markedConstructors =
        constructors
            .stream()
            .filter(c -> c.getAnnotation(AutoCodec.Constructor.class) != null)
            .collect(toImmutableList());
    ExecutableElement constructor = null;
    if (markedConstructors.isEmpty()) {
      // If nothing is marked, see if there is a unique constructor.
      if (constructors.size() > 1) {
        throw new IllegalArgumentException(
            encodedType.getQualifiedName()
                + " has multiple constructors but no Constructor annotation.");
      }
      // In Java, every class has at least one constructor, so this never fails.
      constructor = constructors.get(0);
    } else if (markedConstructors.size() == 1) {
      constructor = markedConstructors.get(0);
    } else {
      throw new IllegalArgumentException(
          encodedType.getQualifiedName() + " has multiple Constructor annotations.");
    }
    List<? extends VariableElement> constructorParameters = constructor.getParameters();
    initializeUnsafeOffsets(codecClassBuilder, encodedType, constructorParameters);
    codecClassBuilder.addMethod(
        buildSerializeMethodWithConstructor(encodedType, constructorParameters));
    MethodSpec.Builder deserializeBuilder =
        AutoCodecUtil.initializeDeserializeMethodBuilder(encodedType);
    buildDeserializeBody(deserializeBuilder, constructorParameters);
    addReturnNew(deserializeBuilder, encodedType, constructor, constructorParameters);
    codecClassBuilder.addMethod(deserializeBuilder.build());
  }

  private MethodSpec buildSerializeMethodWithConstructor(
      TypeElement encodedType, List<? extends VariableElement> parameters) {
    MethodSpec.Builder serializeBuilder =
        AutoCodecUtil.initializeSerializeMethodBuilder(encodedType);
    for (VariableElement parameter : parameters) {
      VariableElement field = getFieldByName(encodedType, parameter.getSimpleName().toString());
      TypeKind typeKind = field.asType().getKind();
      switch (typeKind) {
        case BOOLEAN:
          serializeBuilder.addStatement(
              "codedOut.writeBoolNoTag($T.getInstance().getBoolean(input, $L_offset))",
              UnsafeProvider.class,
              parameter.getSimpleName());
          break;
        case INT:
          serializeBuilder.addStatement(
              "codedOut.writeInt32NoTag($T.getInstance().getInt(input, $L_offset))",
              UnsafeProvider.class,
              parameter.getSimpleName());
          break;
        case DECLARED:
          serializeBuilder.addStatement(
              "$T unsafe_$L = ($T)$T.getInstance().getObject(input, $L_offset)",
              field.asType(),
              parameter.getSimpleName(),
              field.asType(),
              UnsafeProvider.class,
              parameter.getSimpleName());
          marshallers.writeSerializationCode(
              new Marshaller.Context(
                  serializeBuilder,
                  (DeclaredType) parameter.asType(),
                  "unsafe_" + parameter.getSimpleName()));
          break;
        default:
          throw new UnsupportedOperationException("Unimplemented or invalid kind: " + typeKind);
      }
    }
    return serializeBuilder.build();
  }

  private void buildClassWithPublicFieldsStrategy(
      TypeSpec.Builder codecClassBuilder, TypeElement encodedType) {
    List<? extends VariableElement> publicFields =
        ElementFilter.fieldsIn(env.getElementUtils().getAllMembers(encodedType))
            .stream()
            .filter(this::isPublicField)
            .collect(Collectors.toList());
    codecClassBuilder.addMethod(buildSerializeMethodWithPublicFields(encodedType, publicFields));
    MethodSpec.Builder deserializeBuilder =
        AutoCodecUtil.initializeDeserializeMethodBuilder(encodedType);
    buildDeserializeBody(deserializeBuilder, publicFields);
    addInstantiatePopulateFieldsAndReturn(deserializeBuilder, encodedType, publicFields);
    codecClassBuilder.addMethod(deserializeBuilder.build());
  }

  private boolean isPublicField(VariableElement element) {
    if (matchesType(element.asType(), Void.class)) {
      return false; // Void types can't be instantiated, so the processor ignores them completely.
    }
    Set<Modifier> modifiers = element.getModifiers();
    return modifiers.contains(Modifier.PUBLIC) && !modifiers.contains(Modifier.STATIC);
  }

  private MethodSpec buildSerializeMethodWithPublicFields(
      TypeElement encodedType, List<? extends VariableElement> parameters) {
    MethodSpec.Builder serializeBuilder =
        AutoCodecUtil.initializeSerializeMethodBuilder(encodedType);
    for (VariableElement parameter : parameters) {
      String paramAccessor = "input." + parameter.getSimpleName();
      TypeKind typeKind = parameter.asType().getKind();
      switch (typeKind) {
        case BOOLEAN:
          serializeBuilder.addStatement("codedOut.writeBoolNoTag($L)", paramAccessor);
          break;
        case INT:
          serializeBuilder.addStatement("codedOut.writeInt32NoTag($L)", paramAccessor);
          break;
        case DECLARED:
          marshallers.writeSerializationCode(
              new Marshaller.Context(
                  serializeBuilder, (DeclaredType) parameter.asType(), paramAccessor));
          break;
        default:
          throw new UnsupportedOperationException("Unimplemented or invalid kind: " + typeKind);
      }
    }
    return serializeBuilder.build();
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
          marshallers.writeDeserializationCode(
              new Marshaller.Context(builder, (DeclaredType) parameter.asType(), paramName));
          break;
        default:
          throw new IllegalArgumentException("Unimplemented or invalid kind: " + typeKind);
      }
    }
  }

  /**
   * Invokes the constructor and returns the value.
   *
   * <p>Used by the {@link AutoCodec.Strategy.CONSTRUCTOR} strategy.
   */
  private static void addReturnNew(
      MethodSpec.Builder builder,
      TypeElement type,
      ExecutableElement constructor,
      List<? extends VariableElement> parameters) {
    List<? extends TypeMirror> allThrown = constructor.getThrownTypes();
    if (!allThrown.isEmpty()) {
      builder.beginControlFlow("try");
    }
    builder.addStatement(
        "return new $T($L)",
        TypeName.get(type.asType()),
        parameters.stream().map(p -> p.getSimpleName() + "_").collect(Collectors.joining(", ")));
    if (!allThrown.isEmpty()) {
      for (TypeMirror thrown : allThrown) {
        builder.nextControlFlow("catch ($T e)", TypeName.get(thrown));
        builder.addStatement(
            "throw new $T(\"$L constructor threw an exception\", e)",
            SerializationException.class,
            type.getQualifiedName());
      }
      builder.endControlFlow();
    }
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

  /**
   * Adds fields to the codec class to hold offsets and adds a constructor to initialize them.
   *
   * <p>For a parameter with name {@code target}, the field will have name {@code target_offset}.
   *
   * @param parameters constructor parameters
   */
  private void initializeUnsafeOffsets(
      TypeSpec.Builder builder,
      TypeElement encodedType,
      List<? extends VariableElement> parameters) {
    MethodSpec.Builder constructor = MethodSpec.constructorBuilder();
    for (VariableElement param : parameters) {
      VariableElement field = getFieldByName(encodedType, param.getSimpleName().toString());
      if (!env.getTypeUtils().isSameType(field.asType(), param.asType())) {
        throw new IllegalArgumentException(
            encodedType.getQualifiedName()
                + " field "
                + field.getSimpleName()
                + " has mismatching type.");
      }
      builder.addField(
          TypeName.LONG, param.getSimpleName() + "_offset", Modifier.PRIVATE, Modifier.FINAL);
      constructor.beginControlFlow("try");
      // TODO(shahan): also support fields defined in superclasses if needed.
      constructor.addStatement(
          "this.$L_offset = $T.getInstance().objectFieldOffset($T.class.getDeclaredField(\"$L\"))",
          param.getSimpleName(),
          UnsafeProvider.class,
          encodedType.asType(),
          param.getSimpleName());
      constructor.nextControlFlow("catch ($T e)", NoSuchFieldException.class);
      constructor.addStatement("throw new $T(e)", IllegalStateException.class);
      constructor.endControlFlow();
    }
    builder.addMethod(constructor.build());
  }

  /**
   * Returns the VariableElement for the field named {@code name}.
   *
   * <p>Throws IllegalArgumentException if no such field is found.
   */
  private VariableElement getFieldByName(TypeElement type, String name) {
    return ElementFilter.fieldsIn(type.getEnclosedElements())
        .stream()
        .filter(f -> f.getSimpleName().contentEquals(name))
        .findAny()
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    type.getQualifiedName() + ": no field with name matching " + name));
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
    builder.addStatement(
        "throw new $T(input.getClass().getName(), e)", SerializationException.class);
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
    builder.addStatement("throw new $T(className, e)", SerializationException.class);
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

  /** Emits a note to BUILD log during annotation processing for debugging. */
  private void note(String note) {
    env.getMessager().printMessage(Diagnostic.Kind.NOTE, note);
  }
}
