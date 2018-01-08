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
import com.google.devtools.build.lib.skyframe.serialization.PolymorphicHelper;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
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
import javax.lang.model.type.MirroredTypeException;
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
      @Nullable TypeElement dependencyType = getDependencyType(annotation);
      TypeSpec.Builder codecClassBuilder = null;
      switch (annotation.strategy()) {
        case CONSTRUCTOR:
          codecClassBuilder = buildClassWithConstructorStrategy(encodedType, dependencyType);
          break;
        case PUBLIC_FIELDS:
          codecClassBuilder = buildClassWithPublicFieldsStrategy(encodedType, dependencyType);
          break;
        case POLYMORPHIC:
          codecClassBuilder = buildClassWithPolymorphicStrategy(encodedType, dependencyType);
          break;
        default:
          throw new IllegalArgumentException("Unknown strategy: " + annotation.strategy());
      }
      codecClassBuilder.addMethod(
          AutoCodecUtil.initializeGetEncodedClassMethod(encodedType)
              .addStatement("return $T.class", TypeName.get(encodedType.asType()))
              .build());
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

  /** Returns the type of the annotation dependency or null if the type is {@link Void}. */
  @Nullable
  private TypeElement getDependencyType(AutoCodec annotation) {
    try {
      annotation.dependency();
      throw new AssertionError("Expected MirroredTypeException!");
    } catch (MirroredTypeException e) {
      DeclaredType dependencyMirror = (DeclaredType) e.getTypeMirror();
      if (matchesType(dependencyMirror, Void.class)) {
        return null;
      }
      return (TypeElement) dependencyMirror.asElement();
    }
  }

  private TypeSpec.Builder buildClassWithConstructorStrategy(
      TypeElement encodedType, @Nullable TypeElement dependency) {
    ExecutableElement constructor = selectConstructorForConstructorStrategy(encodedType);
    PartitionedParameters parameters = isolateDependency(constructor);
    if (dependency != null) {
      if (parameters.dependency != null) {
        throw new IllegalArgumentException(
            encodedType.getQualifiedName()
                + " has both a @Dependency annotated constructor parameter "
                + "and a non-Void dependency element "
                + dependency.getQualifiedName());
      }
      parameters.dependency = dependency;
    }

    TypeSpec.Builder codecClassBuilder =
        AutoCodecUtil.initializeCodecClassBuilder(encodedType, parameters.dependency);

    initializeUnsafeOffsets(codecClassBuilder, encodedType, parameters.fields);

    codecClassBuilder.addMethod(buildSerializeMethodWithConstructor(encodedType, parameters));

    MethodSpec.Builder deserializeBuilder =
        AutoCodecUtil.initializeDeserializeMethodBuilder(encodedType, parameters.dependency);
    buildDeserializeBody(deserializeBuilder, parameters.fields);
    addReturnNew(deserializeBuilder, encodedType, constructor);
    codecClassBuilder.addMethod(deserializeBuilder.build());

    return codecClassBuilder;
  }

  private static class PartitionedParameters {
    /** Non-dependency parameters. */
    List<VariableElement> fields;
    /** Dependency for this codec or null if no such dependency exists. */
    @Nullable TypeElement dependency;
  }

  /** Separates any dependency from the constructor parameters. */
  private static PartitionedParameters isolateDependency(ExecutableElement constructor) {
    Map<Boolean, List<VariableElement>> splitParameters =
        constructor
            .getParameters()
            .stream()
            .collect(
                Collectors.partitioningBy(
                    p -> p.getAnnotation(AutoCodec.Dependency.class) != null));
    PartitionedParameters result = new PartitionedParameters();
    result.fields = splitParameters.get(Boolean.FALSE);
    List<VariableElement> dependencies = splitParameters.get(Boolean.TRUE);
    if (dependencies.size() > 1) {
      throw new IllegalArgumentException(
          ((TypeElement) constructor.getEnclosingElement()).getQualifiedName()
              + " constructor has multiple Dependency annotations.");
    }
    if (!dependencies.isEmpty()) {
      result.dependency = (TypeElement) ((DeclaredType) dependencies.get(0).asType()).asElement();
    }
    return result;
  }

  private static ExecutableElement selectConstructorForConstructorStrategy(
      TypeElement encodedType) {
    List<ExecutableElement> constructors =
        ElementFilter.constructorsIn(encodedType.getEnclosedElements());
    ImmutableList<ExecutableElement> markedConstructors =
        constructors
            .stream()
            .filter(c -> c.getAnnotation(AutoCodec.Constructor.class) != null)
            .collect(toImmutableList());
    if (markedConstructors.isEmpty()) {
      // If nothing is marked, see if there is a unique constructor.
      if (constructors.size() > 1) {
        throw new IllegalArgumentException(
            encodedType.getQualifiedName()
                + " has multiple constructors but no Constructor annotation.");
      }
      // In Java, every class has at least one constructor, so this never fails.
      return constructors.get(0);
    }
    if (markedConstructors.size() == 1) {
      return markedConstructors.get(0);
    }
    throw new IllegalArgumentException(
        encodedType.getQualifiedName() + " has multiple Constructor annotations.");
  }

  private MethodSpec buildSerializeMethodWithConstructor(
      TypeElement encodedType, PartitionedParameters parameters) {
    MethodSpec.Builder serializeBuilder =
        AutoCodecUtil.initializeSerializeMethodBuilder(encodedType, parameters.dependency);
    for (VariableElement parameter : parameters.fields) {
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

  private TypeSpec.Builder buildClassWithPublicFieldsStrategy(
      TypeElement encodedType, @Nullable TypeElement dependency) {
    TypeSpec.Builder codecClassBuilder =
        AutoCodecUtil.initializeCodecClassBuilder(encodedType, dependency);
    ImmutableList<? extends VariableElement> publicFields =
        ElementFilter.fieldsIn(env.getElementUtils().getAllMembers(encodedType))
            .stream()
            .filter(this::isPublicField)
            .collect(toImmutableList());
    codecClassBuilder.addMethod(
        buildSerializeMethodWithPublicFields(encodedType, publicFields, dependency));
    MethodSpec.Builder deserializeBuilder =
        AutoCodecUtil.initializeDeserializeMethodBuilder(encodedType, dependency);
    buildDeserializeBody(deserializeBuilder, publicFields);
    addInstantiatePopulateFieldsAndReturn(deserializeBuilder, encodedType, publicFields);
    codecClassBuilder.addMethod(deserializeBuilder.build());
    return codecClassBuilder;
  }

  private boolean isPublicField(VariableElement element) {
    if (matchesType(element.asType(), Void.class)) {
      return false; // Void types can't be instantiated, so the processor ignores them completely.
    }
    Set<Modifier> modifiers = element.getModifiers();
    return modifiers.contains(Modifier.PUBLIC) && !modifiers.contains(Modifier.STATIC);
  }

  private MethodSpec buildSerializeMethodWithPublicFields(
      TypeElement encodedType,
      List<? extends VariableElement> parameters,
      @Nullable TypeElement dependency) {
    MethodSpec.Builder serializeBuilder =
        AutoCodecUtil.initializeSerializeMethodBuilder(encodedType, dependency);
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
      MethodSpec.Builder builder, TypeElement type, ExecutableElement constructor) {
    List<? extends TypeMirror> allThrown = constructor.getThrownTypes();
    if (!allThrown.isEmpty()) {
      builder.beginControlFlow("try");
    }
    String parameters =
        constructor
            .getParameters()
            .stream()
            .map(AutoCodecProcessor::handleFromParameter)
            .collect(Collectors.joining(", "));
    builder.addStatement("return new $T($L)", TypeName.get(type.asType()), parameters);
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
   * Coverts a constructor parameter to a String representing its handle within deserialize.
   *
   * <p>Uses the handle {@code dependency} for any parameter with the {@link AutoCodec.Dependency}
   * annotation.
   */
  private static String handleFromParameter(VariableElement parameter) {
    return parameter.getAnnotation(AutoCodec.Dependency.class) != null
        ? "dependency"
        : (parameter.getSimpleName() + "_");
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
  private static VariableElement getFieldByName(TypeElement type, String name) {
    return ElementFilter.fieldsIn(type.getEnclosedElements())
        .stream()
        .filter(f -> f.getSimpleName().contentEquals(name))
        .findAny()
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    type.getQualifiedName() + ": no field with name matching " + name));
  }

  private static TypeSpec.Builder buildClassWithPolymorphicStrategy(
      TypeElement encodedType, @Nullable TypeElement dependency) {
    if (!encodedType.getModifiers().contains(Modifier.ABSTRACT)) {
      throw new IllegalArgumentException(
          encodedType + " is not abstract, but POLYMORPHIC was selected as the strategy.");
    }
    TypeSpec.Builder codecClassBuilder =
        AutoCodecUtil.initializeCodecClassBuilder(encodedType, dependency);
    codecClassBuilder.addMethod(buildPolymorphicSerializeMethod(encodedType, dependency));
    codecClassBuilder.addMethod(buildPolymorphicDeserializeMethod(encodedType, dependency));
    return codecClassBuilder;
  }

  private static MethodSpec buildPolymorphicSerializeMethod(
      TypeElement encodedType, @Nullable TypeElement dependency) {
    MethodSpec.Builder builder =
        AutoCodecUtil.initializeSerializeMethodBuilder(encodedType, dependency);
    if (dependency == null) {
      builder.addStatement("$T.serialize(input, codedOut, null)", PolymorphicHelper.class);
    } else {
      builder.addStatement(
          "$T.serialize(input, codedOut, $T.ofNullable(dependency))",
          PolymorphicHelper.class,
          Optional.class);
    }
    return builder.build();
  }

  private static MethodSpec buildPolymorphicDeserializeMethod(
      TypeElement encodedType, @Nullable TypeElement dependency) {
    MethodSpec.Builder builder =
        AutoCodecUtil.initializeDeserializeMethodBuilder(encodedType, dependency);
    if (dependency == null) {
      builder.addStatement(
          "return ($T) $T.deserialize(codedIn, null)",
          TypeName.get(encodedType.asType()),
          PolymorphicHelper.class);
    } else {
      builder.addStatement(
          "return ($T) $T.deserialize(codedIn, $T.ofNullable(dependency))",
          TypeName.get(encodedType.asType()),
          PolymorphicHelper.class,
          Optional.class);
    }
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
