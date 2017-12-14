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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.ParameterizedTypeName;
import com.squareup.javapoet.TypeName;
import com.squareup.javapoet.TypeSpec;
import java.io.IOException;
import java.util.Comparator;
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
  // Synthesized classes will be prefixed with AutoCodec_.
  public static final String GENERATED_CLASS_NAME_PREFIX = "AutoCodec";
  private static final Class<AutoCodec> ANNOTATION = AutoCodec.class;

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
    return ImmutableSet.of(ANNOTATION.getCanonicalName());
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
    for (Element element : roundEnv.getElementsAnnotatedWith(ANNOTATION)) {
      AutoCodec annotation = element.getAnnotation(ANNOTATION);
      switch (annotation.strategy()) {
        case CONSTRUCTOR:
          buildCodecUsingConstructor((TypeElement) element);
          break;
        default:
          throw new IllegalArgumentException("Unknown strategy: " + annotation.strategy());
      }
    }
    return true;
  }

  /**
   * Uses the first constructor of the class to synthesize a codec.
   *
   * <p>This strategy depends on
   *
   * <ul>
   *   <li>the class constructor taking all serialized fields as parameters
   *   <li>and each serialized field having a corresponding getter.
   * </ul>
   *
   * For example, a constructor having parameter, {@code target}, should having a matching getter,
   * {@code getTarget()}.
   *
   * <p>The first constructor is the first ocurring in the source code.
   */
  private void buildCodecUsingConstructor(TypeElement classElement) {
    TypeSpec.Builder codecClassBuilder =
        TypeSpec.classBuilder(getCodecName(classElement))
            .superclass(TypeName.get(classElement.asType()));

    TypeElement encodedType = getEncodedType(classElement);

    // Generates the getEncodedClass method.
    codecClassBuilder.addMethod(
        MethodSpec.methodBuilder("getEncodedClass")
            .addModifiers(Modifier.PUBLIC)
            .addAnnotation(Override.class)
            .returns(
                ParameterizedTypeName.get(
                    ClassName.get(Class.class), TypeName.get(encodedType.asType())))
            .addStatement("return $T.class", TypeName.get(encodedType.asType()))
            .build());

    // In Java, every class has a constructor, so this always succeeds.
    ExecutableElement constructor =
        ElementFilter.constructorsIn(encodedType.getEnclosedElements()).get(0);
    List<? extends VariableElement> constructorParameters = constructor.getParameters();
    addSerializeMethodUsingConstructor(codecClassBuilder, encodedType, constructorParameters);
    addDeserializeMethodUsingConstructor(codecClassBuilder, encodedType, constructorParameters);

    String packageName =
        env.getElementUtils().getPackageOf(classElement).getQualifiedName().toString();
    try {
      JavaFile file = JavaFile.builder(packageName, codecClassBuilder.build()).build();
      file.writeTo(env.getFiler());
      if (env.getOptions().containsKey("autocodec_print_generated")) {
        note("AutoCodec generated codec for " + classElement + ":\n" + file);
      }
    } catch (IOException e) {
      env.getMessager()
          .printMessage(Diagnostic.Kind.ERROR, "Failed to generate output file: " + e.getMessage());
    }
  }

  /**
   * Heuristic that converts a constructor parameter to a getter.
   *
   * <p>For example, a parameter called {@code target} results in {@code getTarget()}.
   */
  private static String paramNameAsAccessor(String name) {
    return "get" + name.substring(0, 1).toUpperCase() + name.substring(1) + "()";
  }

  /**
   * Name of the generated codec class.
   *
   * <p>For {@code Foo.Bar.Codec} this is {@code AutoCodec_Foo_Bar_Codec}.
   */
  private static String getCodecName(Element element) {
    ImmutableList.Builder<String> classNamesBuilder = new ImmutableList.Builder<>();
    do {
      classNamesBuilder.add(element.getSimpleName().toString());
      element = element.getEnclosingElement();
    } while (element instanceof TypeElement);
    classNamesBuilder.add(GENERATED_CLASS_NAME_PREFIX);
    return classNamesBuilder.build().reverse().stream().collect(Collectors.joining("_"));
  }

  private void addSerializeMethodUsingConstructor(
      TypeSpec.Builder codecClassBuilder,
      TypeElement encodedType,
      List<? extends VariableElement> constructorParameters) {
    MethodSpec.Builder serializeBuilder =
        MethodSpec.methodBuilder("serialize")
            .addModifiers(Modifier.PUBLIC)
            .returns(void.class)
            .addParameter(TypeName.get(encodedType.asType()), "input")
            .addParameter(CodedOutputStream.class, "codedOut")
            .addAnnotation(Override.class)
            .addException(SerializationException.class)
            .addException(IOException.class);
    for (VariableElement parameter : constructorParameters) {
      buildSerializeBody(
          serializeBuilder,
          (DeclaredType) parameter.asType(),
          "input." + paramNameAsAccessor(parameter.getSimpleName().toString()));
    }
    codecClassBuilder.addMethod(serializeBuilder.build());
  }

  private void addDeserializeMethodUsingConstructor(
      TypeSpec.Builder codecClassBuilder,
      TypeElement encodedType,
      List<? extends VariableElement> constructorParameters) {
    MethodSpec.Builder deserializeBuilder =
        MethodSpec.methodBuilder("deserialize")
            .addModifiers(Modifier.PUBLIC)
            .returns(TypeName.get(encodedType.asType()))
            .addParameter(CodedInputStream.class, "codedIn")
            .addAnnotation(Override.class)
            .addException(SerializationException.class)
            .addException(IOException.class);
    for (VariableElement parameter : constructorParameters) {
      buildDeserializeBody(
          deserializeBuilder,
          (DeclaredType) parameter.asType(),
          parameter.getSimpleName().toString());
    }
    // Invokes the constructor and returns the value.
    deserializeBuilder.addStatement(
        "return new $T($L)",
        TypeName.get(encodedType.asType()),
        constructorParameters
            .stream()
            .map(p -> p.getSimpleName().toString())
            .collect(Collectors.joining(", ")));
    codecClassBuilder.addMethod(deserializeBuilder.build());
  }

  /**
   * Appends code statements to {@code builder} to serialize a pre-declared variable named {@code
   * accessor}.
   *
   * @param type the type of {@code accessor}
   */
  private void buildSerializeBody(MethodSpec.Builder builder, DeclaredType type, String accessor) {
    builder.beginControlFlow("if ($L != null)", accessor); // Begin if not null block.
    builder.addStatement("codedOut.writeBoolNoTag(true)");
    // TODO(shahan): Add support for more types.
    if (matchesErased(type, ImmutableSortedSet.class)) {
      // Writes the target count to the stream so deserialization knows when to stop.
      builder.addStatement("codedOut.writeInt32NoTag($L.size())", accessor);
      DeclaredType repeatedType = (DeclaredType) type.getTypeArguments().get(0);
      // TODO(shahan): consider introducing a depth parameter to avoid shadowing here.
      builder.beginControlFlow("for ($T repeated : $L)", TypeName.get(repeatedType), accessor);
      buildSerializeBody(builder, repeatedType, "repeated");
      builder.endControlFlow();
    } else {
      // Otherwise use the type's CODEC.
      builder.addStatement("$T.CODEC.serialize($L, codedOut)", TypeName.get(type), accessor);
    }
    builder.nextControlFlow("else");
    builder.addStatement("codedOut.writeBoolNoTag(false)");
    builder.endControlFlow(); // End if not null.
  }

  /**
   * Appends code statements to {@code builder} declaring a variable called {@code name} and
   * initializing it by deserialization.
   *
   * @param type the type of {@code name}
   */
  private void buildDeserializeBody(MethodSpec.Builder builder, DeclaredType type, String name) {
    builder.addStatement("$T $L = null", TypeName.get(type), name);
    builder.beginControlFlow("if (codedIn.readBool())"); // Begin null-handling block.
    // TODO(shahan): Add support for more types.
    if (matchesErased(type, ImmutableSortedSet.class)) {
      DeclaredType repeatedType = (DeclaredType) type.getTypeArguments().get(0);
      builder.addStatement(
          "$T<$T> builder = new $T<>($T.naturalOrder())",
          ImmutableSortedSet.Builder.class,
          TypeName.get(repeatedType),
          ImmutableSortedSet.Builder.class,
          Comparator.class);
      builder.addStatement("int length = codedIn.readInt32()");
      builder.beginControlFlow("for (int i = 0; i < length; ++i)");
      buildDeserializeBody(builder, repeatedType, "repeated");
      builder.addStatement("builder.add(repeated)");
      builder.endControlFlow();
      builder.addStatement("$L = builder.build()", name);
    } else {
      // Otherwise, use the type's CODEC value.
      builder.addStatement("$L = $T.CODEC.deserialize(codedIn)", name, TypeName.get(type));
    }
    builder.endControlFlow(); // End null-handling block.
  }

  /**
   * Gets the type parameter of ObjectCodec, i.e., the type being encoded.
   *
   * <p>{@code element} must implement ObjectCodec.
   */
  private TypeElement getEncodedType(TypeElement element) {
    for (TypeMirror implementedInterface : element.getInterfaces()) {
      if (matchesErased(implementedInterface, ObjectCodec.class)) {
        return (TypeElement)
            env.getTypeUtils()
                .asElement(((DeclaredType) implementedInterface).getTypeArguments().get(0));
      }
    }
    throw new IllegalArgumentException(element + " does not implement ObjectCodec!");
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

  /** Emits a note to BUILD log during annotation processing for debugging. */
  private void note(String note) {
    env.getMessager().printMessage(Diagnostic.Kind.NOTE, note);
  }
}
