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

import static com.google.common.base.Ascii.toLowerCase;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodecProcessor.InstantiatorKind.CONSTRUCTOR;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodecProcessor.InstantiatorKind.FACTORY_METHOD;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodecProcessor.InstantiatorKind.INTERNER;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.findRelationWithGenerics;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getErasure;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.getErasureAsMirror;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.sanitizeTypeParameter;
import static com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.writeGeneratedClassToFile;

import com.google.auto.service.AutoService;
import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Instantiator;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Interner;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationCodeGenerator.Marshaller;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.TypeOperations.Relation;
import com.google.devtools.build.lib.unsafe.UnsafeProvider;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.util.List;
import java.util.Optional;
import java.util.Set;
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
    return ImmutableSet.of(AutoCodec.class.getCanonicalName());
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
    try {
      processInternal(roundEnv);
    } catch (SerializationProcessingException e) {
      // Reporting a message with ERROR kind will fail compilation.
      env.getMessager().printMessage(Diagnostic.Kind.ERROR, e.getMessage(), e.getElement());
    }
    return false;
  }

  private void processInternal(RoundEnvironment roundEnv) throws SerializationProcessingException {
    for (Element element : roundEnv.getElementsAnnotatedWith(AutoCodec.class)) {
      AutoCodec annotation = element.getAnnotation(AutoCodec.class);
      TypeElement encodedType = (TypeElement) element;
      ResolvedInstantiator instantiator = determineInstantiator(encodedType);
      TypeSpec codecClass;
      switch (instantiator.kind()) {
        case CONSTRUCTOR:
        case FACTORY_METHOD:
          codecClass = defineClassWithInstantiator(encodedType, instantiator.method(), annotation);
          break;
        case INTERNER:
          codecClass = defineClassWithInterner(encodedType, instantiator.method(), annotation);
          break;
        default:
          throw new IllegalStateException(
              String.format("Unknown instantiator kind: %s\n", instantiator));
      }

      JavaFile file = writeGeneratedClassToFile(element, codecClass, env);
      if (env.getOptions().containsKey(PRINT_GENERATED_OPTION)) {
        note("AutoCodec generated codec for " + element + ":\n" + file);
      }
    }
  }

  private TypeSpec defineClassWithInstantiator(
      TypeElement encodedType, ExecutableElement instantiator, AutoCodec annotation)
      throws SerializationProcessingException {
    List<? extends VariableElement> fields = instantiator.getParameters();

    TypeSpec.Builder codecClassBuilder =
        Initializers.initializeCodecClassBuilder(encodedType, env)
            .addSuperinterface(
                ParameterizedTypeName.get(
                    ClassName.get(ObjectCodec.class), getErasure(encodedType, env)));

    if (encodedType.getAnnotation(AutoValue.class) == null) {
      initializeUnsafeOffsets(codecClassBuilder, encodedType, fields);
      codecClassBuilder.addMethod(
          buildSerializeMethodWithInstantiator(encodedType, fields, annotation));
    } else {
      codecClassBuilder.addMethod(
          buildSerializeMethodWithInstantiatorForAutoValue(encodedType, fields, annotation));
    }

    MethodSpec.Builder deserializeBuilder =
        Initializers.initializeDeserializeMethodBuilder(encodedType, env);
    buildDeserializeBody(deserializeBuilder, fields);
    addReturnNew(deserializeBuilder, encodedType, instantiator, env);
    codecClassBuilder.addMethod(deserializeBuilder.build());

    return codecClassBuilder.build();
  }

  private TypeSpec defineClassWithInterner(
      TypeElement encodedType, ExecutableElement interner, AutoCodec annotation)
      throws SerializationProcessingException {
    return new InterningObjectCodecGenerator(env).defineCodec(encodedType, annotation, interner);
  }

  enum InstantiatorKind {
    CONSTRUCTOR,
    FACTORY_METHOD,
    INTERNER
  }

  @AutoValue
  abstract static class ResolvedInstantiator {
    public abstract InstantiatorKind kind();

    public abstract ExecutableElement method();

    private static ResolvedInstantiator create(InstantiatorKind kind, ExecutableElement method) {
      return new AutoValue_AutoCodecProcessor_ResolvedInstantiator(kind, method);
    }
  }

  /**
   * Determines the {@link ResolvedInstantiator} by scanning the constructors and methods for
   * annotations.
   *
   * <p>Identifies the {@link Instantiator} or {@link Interner} annotations, throwing an exception
   * if there are multiple. Falls back to checking for a unique constructor or throws otherwise.
   */
  private ResolvedInstantiator determineInstantiator(TypeElement encodedType)
      throws SerializationProcessingException {
    InstantiatorKind instantiatorKind = null;
    ExecutableElement markedMethod = null;

    List<ExecutableElement> constructors =
        ElementFilter.constructorsIn(encodedType.getEnclosedElements());

    for (ExecutableElement constructor : constructors) {
      if (hasInstantiatorAnnotation(constructor)) {
        if (markedMethod != null) {
          throw new SerializationProcessingException(
              encodedType,
              "%s has multiple constructors with the Instantiator annotation.",
              encodedType.getQualifiedName());
        }
        markedMethod = constructor;
        instantiatorKind = CONSTRUCTOR;
      }
    }

    for (ExecutableElement method : ElementFilter.methodsIn(encodedType.getEnclosedElements())) {
      if (hasInstantiatorAnnotation(method)) {
        verifyFactoryMethod(encodedType, method);
        if (markedMethod != null) {
          throw new SerializationProcessingException(
              encodedType,
              "%s has multiple Instantiator or Interner annotations: %s %s.",
              encodedType.getQualifiedName(),
              markedMethod.getSimpleName(),
              method.getSimpleName());
        }
        markedMethod = method;
        instantiatorKind = FACTORY_METHOD;
      }
      if (hasInternerAnnotation(method)) {
        verifyInterner(encodedType, method);
        if (markedMethod != null) {
          throw new SerializationProcessingException(
              encodedType,
              "%s has multiple Instantiator or Interner annotations: %s %s.",
              encodedType.getQualifiedName(),
              markedMethod.getSimpleName(),
              method.getSimpleName());
        }
        markedMethod = method;
        instantiatorKind = INTERNER;
      }
    }

    if (markedMethod != null) {
      return ResolvedInstantiator.create(instantiatorKind, markedMethod);
    }

    // If nothing is marked, see if there is a unique constructor.
    if (constructors.size() > 1) {
      throw new SerializationProcessingException(
          encodedType,
          "%s has multiple constructors but no Instantiator or Interner annotation.",
          encodedType.getQualifiedName());
    }
    // In Java, every class has at least one constructor, so this never fails.
    return ResolvedInstantiator.create(CONSTRUCTOR, constructors.get(0));
  }

  private static boolean hasInstantiatorAnnotation(ExecutableElement elt) {
    return elt.getAnnotation(Instantiator.class) != null;
  }

  private static boolean hasInternerAnnotation(ExecutableElement elt) {
    return elt.getAnnotation(Interner.class) != null;
  }

  private void verifyFactoryMethod(TypeElement encodedType, ExecutableElement elt)
      throws SerializationProcessingException {
    boolean success = elt.getModifiers().contains(Modifier.STATIC);
    if (success) {
      Relation equalityTest =
          findRelationWithGenerics(elt.getReturnType(), encodedType.asType(), env);
      success = equalityTest == Relation.EQUAL_TO || equalityTest == Relation.INSTANCE_OF;
    }
    if (!success) {
      throw new SerializationProcessingException(
          encodedType,
          "%s tags %s as an Instantiator, but it's not a valid factory method %s, %s",
          encodedType,
          elt.getSimpleName(),
          elt.getReturnType(),
          encodedType.asType());
    }
  }

  private void verifyInterner(TypeElement encodedType, ExecutableElement method)
      throws SerializationProcessingException {
    if (!method.getModifiers().contains(Modifier.STATIC)) {
      throw new SerializationProcessingException(
          encodedType, "%s is tagged @Interner, but it's not static.", method.getSimpleName());
    }
    List<? extends VariableElement> parameters = method.getParameters();
    if (parameters.size() != 1) {
      throw new SerializationProcessingException(
          encodedType,
          "%s is tagged @Interner, but it has %d parameters instead of 1.",
          method.getSimpleName(),
          parameters.size());
    }
    TypeMirror subjectType = getErasureAsMirror(encodedType, env);

    // The method should be able to accept a value of encodedType;
    TypeMirror parameterType = getErasureAsMirror(parameters.get(0).asType(), env);
    if (!env.getTypeUtils().isAssignable(subjectType, parameterType)) {
      throw new SerializationProcessingException(
          encodedType,
          "%s is tagged @Interner, but cannot accept a value of type %s because it is not"
              + " assignable to %s.",
          method.getSimpleName(),
          encodedType,
          parameterType);
    }

    // The method should return a value that can be assigned to encodedType.
    TypeMirror returnType = getErasureAsMirror(method.getReturnType(), env);
    if (!env.getTypeUtils().isAssignable(returnType, subjectType)) {
      throw new SerializationProcessingException(
          encodedType,
          "%s is tagged @Interner, but its return type %s cannot be assigned to type %s.",
          method.getSimpleName(),
          method.getReturnType(),
          encodedType);
    }
  }

  private MethodSpec buildSerializeMethodWithInstantiator(
      TypeElement encodedType, List<? extends VariableElement> fields, AutoCodec annotation)
      throws SerializationProcessingException {
    MethodSpec.Builder serializeBuilder =
        Initializers.initializeSerializeMethodBuilder(encodedType, annotation, env);
    for (VariableElement parameter : fields) {
      Optional<FieldValueAndClass> hasField =
          getFieldByNameRecursive(encodedType, parameter.getSimpleName().toString());
      if (hasField.isPresent()) {
        if (findRelationWithGenerics(hasField.get().value.asType(), parameter.asType(), env)
            == Relation.UNRELATED_TO) {
          throw new SerializationProcessingException(
              parameter,
              "%s: parameter %s's type %s is unrelated to corresponding field type %s",
              encodedType.getQualifiedName(),
              parameter.getSimpleName(),
              parameter.asType(),
              hasField.get().value.asType());
        }
        TypeKind typeKind = parameter.asType().getKind();
        serializeBuilder.addStatement(
            "$T unsafe_$L = ($T) $T.unsafe().get$L(obj, $L_offset)",
            sanitizeTypeParameter(parameter.asType(), env),
            parameter.getSimpleName(),
            sanitizeTypeParameter(parameter.asType(), env),
            UnsafeProvider.class,
            typeKind.isPrimitive() ? firstLetterUpper(toLowerCase(typeKind.toString())) : "Object",
            parameter.getSimpleName());
        marshallers.writeSerializationCode(
            new SerializationCodeGenerator.Context(
                serializeBuilder, parameter.asType(), "unsafe_" + parameter.getSimpleName()));
      } else {
        addSerializeParameterWithGetter(encodedType, parameter, serializeBuilder);
      }
    }
    return serializeBuilder.build();
  }

  private String findGetterForClass(VariableElement parameter, TypeElement type)
      throws SerializationProcessingException {
    List<ExecutableElement> methods =
        ElementFilter.methodsIn(env.getElementUtils().getAllMembers(type));

    ImmutableSet.Builder<String> possibleGetterNamesBuilder =
        ImmutableSet.<String>builder().add(parameter.getSimpleName().toString());

    if (parameter.asType().getKind() == TypeKind.BOOLEAN) {
      possibleGetterNamesBuilder.add(
          addCamelCasePrefix(parameter.getSimpleName().toString(), "is"));
    } else {
      possibleGetterNamesBuilder.add(
          addCamelCasePrefix(parameter.getSimpleName().toString(), "get"));
    }
    ImmutableSet<String> possibleGetterNames = possibleGetterNamesBuilder.build();

    for (ExecutableElement element : methods) {
      if (!element.getModifiers().contains(Modifier.STATIC)
          && !element.getModifiers().contains(Modifier.PRIVATE)
          && possibleGetterNames.contains(element.getSimpleName().toString())
          && findRelationWithGenerics(parameter.asType(), element.getReturnType(), env)
              != Relation.UNRELATED_TO) {
        return element.getSimpleName().toString();
      }
    }

    throw new SerializationProcessingException(
        parameter,
        "%s: No getter found corresponding to parameter %s, %s",
        type,
        parameter.getSimpleName(),
        parameter.asType());
  }

  private static String addCamelCasePrefix(String name, String prefix) {
    return prefix + firstLetterUpper(name);
  }

  private static String firstLetterUpper(String str) {
    return Character.toUpperCase(str.charAt(0)) + (str.length() == 1 ? "" : str.substring(1));
  }

  private void addSerializeParameterWithGetter(
      TypeElement encodedType, VariableElement parameter, MethodSpec.Builder serializeBuilder)
      throws SerializationProcessingException {
    String getter = turnGetterIntoExpression(findGetterForClass(parameter, encodedType));
    marshallers.writeSerializationCode(
        new Marshaller.Context(serializeBuilder, parameter.asType(), getter));
  }

  private static String turnGetterIntoExpression(String getterName) {
    return "obj." + getterName + "()";
  }

  private MethodSpec buildSerializeMethodWithInstantiatorForAutoValue(
      TypeElement encodedType, List<? extends VariableElement> fields, AutoCodec annotation)
      throws SerializationProcessingException {
    MethodSpec.Builder serializeBuilder =
        Initializers.initializeSerializeMethodBuilder(encodedType, annotation, env);
    for (VariableElement parameter : fields) {
      addSerializeParameterWithGetter(encodedType, parameter, serializeBuilder);
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
      MethodSpec.Builder builder, List<? extends VariableElement> fields)
      throws SerializationProcessingException {
    for (VariableElement parameter : fields) {
      String paramName = parameter.getSimpleName() + "_";
      marshallers.writeDeserializationCode(
          new Marshaller.Context(builder, parameter.asType(), paramName));
    }
  }

  /** Invokes the instantiator and returns the value. */
  private static void addReturnNew(
      MethodSpec.Builder builder,
      TypeElement type,
      ExecutableElement instantiator,
      ProcessingEnvironment env) {
    List<? extends TypeMirror> allThrown = instantiator.getThrownTypes();
    if (!allThrown.isEmpty()) {
      builder.beginControlFlow("try");
    }
    TypeName typeName = getErasure(type, env);
    String parameters =
        instantiator.getParameters().stream()
            .map(AutoCodecProcessor::handleFromParameter)
            .collect(Collectors.joining(", "));
    if (instantiator.getKind().equals(ElementKind.CONSTRUCTOR)) {
      builder.addStatement("return new $T($L)", typeName, parameters);
    } else { // Otherwise, it's a factory method.
      builder.addStatement("return $T.$L($L)", typeName, instantiator.getSimpleName(), parameters);
    }
    if (!allThrown.isEmpty()) {
      for (TypeMirror thrown : allThrown) {
        builder.nextControlFlow("catch ($T e)", TypeName.get(thrown));
        builder.addStatement(
            "throw new $T(\"$L instantiator threw an exception\", e)",
            SerializationException.class,
            type.getQualifiedName());
      }
      builder.endControlFlow();
    }
  }

  /** Converts a constructor parameter to a String representing its handle within deserialize. */
  private static String handleFromParameter(VariableElement parameter) {
    return parameter.getSimpleName() + "_";
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
      Optional<FieldValueAndClass> field =
          getFieldByNameRecursive(encodedType, param.getSimpleName().toString());
      if (!field.isPresent()) {
        // Will attempt to use a getter for this field instead.
        continue;
      }
      builder.addField(
          TypeName.LONG, param.getSimpleName() + "_offset", Modifier.PRIVATE, Modifier.FINAL);
      constructor.beginControlFlow("try");
      constructor.addStatement(
          "this.$L_offset = $T.unsafe().objectFieldOffset($T.class.getDeclaredField(\"$L\"))",
          param.getSimpleName(),
          UnsafeProvider.class,
          ClassName.get(field.get().declaringClassType),
          param.getSimpleName());
      constructor.nextControlFlow("catch ($T e)", NoSuchFieldException.class);
      constructor.addStatement("throw new $T(e)", IllegalStateException.class);
      constructor.endControlFlow();
    }
    builder.addMethod(constructor.build());
  }

  /** The value of a field, as well as the class that directly declares it. */
  private static class FieldValueAndClass {
    final VariableElement value;
    final TypeElement declaringClassType;

    FieldValueAndClass(VariableElement value, TypeElement declaringClassType) {
      this.value = value;
      this.declaringClassType = declaringClassType;
    }
  }

  private Optional<FieldValueAndClass> getFieldByNameRecursive(TypeElement type, String name) {
    Optional<VariableElement> field =
        ElementFilter.fieldsIn(type.getEnclosedElements()).stream()
            .filter(f -> f.getSimpleName().contentEquals(name))
            .findAny();

    if (field.isPresent()) {
      return Optional.of(new FieldValueAndClass(field.get(), type));
    }
    if (type.getSuperclass().getKind() != TypeKind.NONE) {
      // Applies the erased superclass type so that it can be used in `T.class`.
      return getFieldByNameRecursive(
          (TypeElement)
              env.getTypeUtils().asElement(env.getTypeUtils().erasure(type.getSuperclass())),
          name);
    }
    return Optional.empty();
  }

  /** Emits a note to BUILD log during annotation processing for debugging. */
  private void note(String note) {
    env.getMessager().printMessage(Diagnostic.Kind.NOTE, note);
  }
}
