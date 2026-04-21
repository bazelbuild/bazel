// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options.processor;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsClass;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.ExecutableType;
import javax.lang.model.type.PrimitiveType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.Diagnostic;
import javax.tools.JavaFileObject;

/**
 * An annotation processor that generates an implementation class for options classes annotated with
 * {@link OptionsClass}.
 */
@SupportedAnnotationTypes("com.google.devtools.common.options.OptionsClass")
public final class OptionsClassProcessor extends AbstractProcessor {

  private Types typeUtils;
  private Elements elementUtils;
  private Messager messager;
  private ImmutableMap<TypeMirror, Converter<?>> defaultConverters;
  private ImmutableMap<Class<?>, PrimitiveType> primitiveTypeMap;

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported();
  }

  // This method is necessary because when bootstrapping Bazel, we need to run the option class
  // annotation processor so we need to build it first. But if we simply reference Converters, we
  // also need all of its transitive dependencies, which is a lot. So instead reference it using
  // reflection and report that no default converters are available during bootstrapping.
  @Nullable
  @SuppressWarnings("unchecked") // uses reflection, can't have generic arguments
  private static Map<Class<?>, Converter<?>> getDefaultConverters() {
    Class<?> converters;
    try {
      converters = Class.forName("com.google.devtools.common.options.Converters");
    } catch (ClassNotFoundException e) {
      return null;
    }

    try {
      return (Map<Class<?>, Converter<?>>) converters.getField("DEFAULT_CONVERTERS").get(null);
    } catch (ReflectiveOperationException e) {
      throw new IllegalArgumentException(e);
    }
  }

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);

    typeUtils = processingEnv.getTypeUtils();
    elementUtils = processingEnv.getElementUtils();
    messager = processingEnv.getMessager();

    primitiveTypeMap =
        new ImmutableMap.Builder<Class<?>, PrimitiveType>()
            .put(int.class, typeUtils.getPrimitiveType(TypeKind.INT))
            .put(double.class, typeUtils.getPrimitiveType(TypeKind.DOUBLE))
            .put(boolean.class, typeUtils.getPrimitiveType(TypeKind.BOOLEAN))
            .put(long.class, typeUtils.getPrimitiveType(TypeKind.LONG))
            .buildOrThrow();

    Map<Class<?>, Converter<?>> defaultConverterMap = getDefaultConverters();
    if (defaultConverterMap == null) {
      defaultConverters = null;
      return;
    }
    ImmutableMap.Builder<TypeMirror, Converter<?>> converterMapBuilder =
        new ImmutableMap.Builder<>();

    for (Map.Entry<Class<?>, Converter<?>> entry : defaultConverterMap.entrySet()) {
      Class<?> converterClass = entry.getKey();
      String typeName = converterClass.getCanonicalName();
      TypeElement typeElement = elementUtils.getTypeElement(typeName);
      if (typeElement != null) {
        converterMapBuilder.put(typeElement.asType(), entry.getValue());
      } else {
        if (primitiveTypeMap.containsKey(converterClass)) {
          PrimitiveType primitiveType = primitiveTypeMap.get(converterClass);
          converterMapBuilder
              .put(primitiveType, entry.getValue())
              .put(typeUtils.boxedClass(primitiveType).asType(), entry.getValue());
        }
      }
    }
    defaultConverters = converterMapBuilder.buildOrThrow();
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    for (Element annotatedElement : roundEnv.getElementsAnnotatedWith(OptionsClass.class)) {
      if (annotatedElement.getKind() != ElementKind.CLASS) {
        continue;
      }
      TypeElement typeElement = (TypeElement) annotatedElement;
      generateWrapper(typeElement);
    }
    return false;
  }

  private void generateWrapper(TypeElement typeElement) {
    TypeMirror optionsBase =
        elementUtils.getTypeElement("com.google.devtools.common.options.OptionsBase").asType();
    if (!typeUtils.isAssignable(typeElement.asType(), optionsBase)) {
      messager.printMessage(
          Diagnostic.Kind.ERROR,
          "@Option annotated fields can only be in classes that inherit from OptionsBase.",
          typeElement);
      return;
    }

    String packageName =
        processingEnv.getElementUtils().getPackageOf(typeElement).getQualifiedName().toString();
    String className =
        typeElement.getQualifiedName().toString().substring(packageName.length() + 1);
    String implClassName = className.replace('.', '_') + "Impl";

    record OptionInfo(String fieldType, String capitalizedFieldName, boolean hasSetterInBase) {}
    List<OptionInfo> options = new ArrayList<>();
    boolean hasErrors = false;

    // First pass: collect option info
    for (Element member : processingEnv.getElementUtils().getAllMembers(typeElement)) {
      if (member.getAnnotation(Option.class) == null) {
        continue;
      }
      if (member.getKind() != ElementKind.METHOD) {
        messager.printMessage(
            Diagnostic.Kind.ERROR, "@Option must be on a method in @OptionsClass classes", member);
        hasErrors = true;
        continue;
      }

      ExecutableElement method = (ExecutableElement) member;
      try {
        checkMethodOption(method);
      } catch (OptionProcessorException e) {
        messager.printMessage(Diagnostic.Kind.ERROR, e.getMessage(), e.getElementInError());
        hasErrors = true;
        continue;
      }

      String methodName = method.getSimpleName().toString();
      String fieldType = method.getReturnType().toString();
      String capitalizedFieldName = methodName.substring("get".length());

      ExecutableElement setter = null;
      String setterName = "set" + capitalizedFieldName;
      for (Element e : processingEnv.getElementUtils().getAllMembers(typeElement)) {
        if (e.getKind() == ElementKind.METHOD && e.getSimpleName().contentEquals(setterName)) {
          setter = (ExecutableElement) e;
          break;
        }
      }

      if (setter != null) {
        if (!setter.getModifiers().contains(Modifier.ABSTRACT)) {
          messager.printMessage(Diagnostic.Kind.ERROR, "Setter must be abstract", setter);
          hasErrors = true;
          continue;
        }

        if (setter.getParameters().size() != 1) {
          messager.printMessage(
              Diagnostic.Kind.ERROR, "Setter must have exactly one argument", setter);
          hasErrors = true;
          continue;
        }

        if (!processingEnv
            .getTypeUtils()
            .isSameType(setter.getParameters().get(0).asType(), method.getReturnType())) {
          messager.printMessage(
              Diagnostic.Kind.ERROR,
              String.format(
                  "Setter argument type must be same as getter return type (%s)", fieldType),
              setter);
          hasErrors = true;
          continue;
        }
      }

      options.add(new OptionInfo(fieldType, capitalizedFieldName, setter != null));
    }

    if (hasErrors) {
      return;
    }

    try {
      // Generate the Impl class
      JavaFileObject implJfo =
          processingEnv.getFiler().createSourceFile(packageName + "." + implClassName);
      try (PrintWriter out = new PrintWriter(implJfo.openWriter())) {
        out.printf(
            """
            package %1$s;

            public class %2$s extends %3$s {
              public %2$s() {
                super();
              }

              @Override
              @SuppressWarnings("unchecked")
              public Class<? extends %3$s> getOptionsClass() {
                return (Class<? extends %3$s>) %3$s.class;
              }
            """,
            packageName, implClassName, typeElement.getQualifiedName());

        for (OptionInfo option : options) {
          String fieldName =
              option.capitalizedFieldName.substring(0, 1).toLowerCase(Locale.ROOT)
                  + option.capitalizedFieldName.substring(1);
          out.printf(
              """
                private %1$s %2$s;

                @Override
                public %1$s get%3$s() {
                  return this.%2$s;
                }

                %4$s
                public void set%3$s(%1$s %2$s) {
                  this.%2$s = %2$s;
                }
              """,
              option.fieldType,
              fieldName,
              option.capitalizedFieldName,
              option.hasSetterInBase ? "@Override" : "");
        }

        out.println("}");
      }
    } catch (IOException e) {
      messager.printMessage(
          Diagnostic.Kind.ERROR,
          "Failed to generate implementation for " + className + ": " + e.getMessage());
    }
  }

  private void checkMethodOption(ExecutableElement method) throws OptionProcessorException {
    if (!method.getModifiers().contains(Modifier.PUBLIC)) {
      throw new OptionProcessorException(method, "@Option method must be public");
    }
    if (!method.getModifiers().contains(Modifier.ABSTRACT)) {
      throw new OptionProcessorException(method, "@Option method must be abstract");
    }

    String methodName = method.getSimpleName().toString();
    if (!methodName.startsWith("get")
        || methodName.length() < 4
        || !Character.isUpperCase(methodName.charAt(3))) {
      throw new OptionProcessorException(
          method, "Annotated method name must start with 'get' followed by an uppercase letter");
    }

    checkOptionName(method);
    checkOldCategoriesAreNotUsed(method);
    checkExpansionOptions(method);
    checkConverter(method);
    checkEffectTagRationality(method);
    checkMetadataTagAndCategoryRationality(method);
    checkNoDefaultValueForMultipleOption(method);
    checkDeprecated(method);
  }

  private void checkOptionName(ExecutableElement method) throws OptionProcessorException {
    Option annotation = method.getAnnotation(Option.class);
    String optionName = annotation.name();
    if (optionName.isEmpty()) {
      throw new OptionProcessorException(method, "Option must have an actual name.");
    }

    if (!ImmutableList.copyOf(annotation.metadataTags()).contains(OptionMetadataTag.INTERNAL)) {
      if (!Pattern.matches("([\\w:-])*", optionName)) {
        // Ideally, this would be just \w, but - and : are needed for legacy options. We can lie in
        // the error though, no harm in encouraging good behavior.
        throw new OptionProcessorException(
            method,
            "Options that are used on the command line as flags must have names made from word "
                + "characters only.");
      }
    }
  }

  private void checkEffectTagRationality(ExecutableElement method) throws OptionProcessorException {
    Option annotation = method.getAnnotation(Option.class);
    OptionEffectTag[] effectTags = annotation.effectTags();
    if (effectTags.length < 1) {
      throw new OptionProcessorException(
          method,
          "Option does not list at least one OptionEffectTag. If the option has no effect, "
              + "please be explicit and add NO_OP. Otherwise, add a tag representing its effect.");
    } else if (effectTags.length > 1) {
      // If there are more than 1 tag, make sure that NO_OP and UNKNOWN is not one of them.
      // These don't make sense if other effects are listed.
      ImmutableList<OptionEffectTag> tags = ImmutableList.copyOf(effectTags);
      if (tags.contains(OptionEffectTag.UNKNOWN)) {
        throw new OptionProcessorException(
            method,
            "Option includes UNKNOWN with other, known, effects. Please remove UNKNOWN from "
                + "the list.");
      }
      if (tags.contains(OptionEffectTag.NO_OP)) {
        throw new OptionProcessorException(
            method,
            "Option includes NO_OP with other effects. This doesn't make much sense. Please "
                + "remove NO_OP or the actual effects from the list, whichever is correct.");
      }
    }
  }

  private void checkMetadataTagAndCategoryRationality(ExecutableElement method)
      throws OptionProcessorException {
    Option annotation = method.getAnnotation(Option.class);
    OptionMetadataTag[] metadataTags = annotation.metadataTags();
    OptionDocumentationCategory category = annotation.documentationCategory();

    for (OptionMetadataTag tag : metadataTags) {
      if (tag == OptionMetadataTag.HIDDEN || tag == OptionMetadataTag.INTERNAL) {
        if (category != OptionDocumentationCategory.UNDOCUMENTED) {
          throw new OptionProcessorException(
              method,
              "Option has metadata tag %s but does not have category UNDOCUMENTED. Please fix.",
              tag);
        }
      }
    }
  }

  private static final ImmutableSet<String> DEPRECATED_CATEGORIES =
      ImmutableSet.of("undocumented", "hidden", "internal");

  private void checkOldCategoriesAreNotUsed(ExecutableElement method)
      throws OptionProcessorException {
    Option annotation = method.getAnnotation(Option.class);
    if (DEPRECATED_CATEGORIES.contains(annotation.category())) {
      throw new OptionProcessorException(
          method,
          "Documentation level is no longer read from the option category. Category \""
              + annotation.category()
              + "\" is disallowed, see OptionMetadataTags for the relevant tags.");
    }
  }

  private void checkExpansionOptions(ExecutableElement method) throws OptionProcessorException {
    Option annotation = method.getAnnotation(Option.class);
    boolean isExpansion = annotation.expansion().length > 0;
    boolean hasImplicitRequirements = annotation.implicitRequirements().length > 0;

    if (isExpansion && hasImplicitRequirements) {
      throw new OptionProcessorException(
          method,
          "Can't set an option to be both an expansion option and have implicit requirements.");
    }

    if (isExpansion || hasImplicitRequirements) {
      if (annotation.allowMultiple()) {
        throw new OptionProcessorException(
            method,
            "Can't set an option to accumulate multiple values and let it expand to other flags.");
      }
    }
  }

  private void checkNoDefaultValueForMultipleOption(ExecutableElement method)
      throws OptionProcessorException {
    Option annotation = method.getAnnotation(Option.class);
    if (annotation.allowMultiple()
        && !annotation.defaultValue().equals("null")
        && !ImmutableList.of("runs_per_test", "flaky_test_attempts").contains(annotation.name())) {
      throw new OptionProcessorException(
          method,
          "Default values for multiple options are not allowed - use \"null\" special value");
    }
  }

  // TODO(Silic0nS0ldier): Remove this allowlist once all tests have been fixed.
  private static final ImmutableList<String> NO_OP_OPTION_ALLOWLIST =
      ImmutableList.of(
          "com.google.devtools.build.lib.analysis.AnalysisCachingTest.",
          "com.google.devtools.build.lib.analysis.config.BuildOptionDetailsTest.",
          "com.google.devtools.build.lib.analysis.config.BuildOptionsTest.",
          "com.google.devtools.build.lib.analysis.LateBoundSplitUtil.",
          "com.google.devtools.build.lib.analysis.producers.BuildConfigurationKeyMapProducerTest.",
          "com.google.devtools.build.lib.analysis.producers.BuildConfigurationKeyProducerTest.",
          "com.google.devtools.build.lib.analysis.RequiredConfigFragmentsTest.",
          "com.google.devtools.build.lib.analysis.starlark.StarlarkTransitionTest.",
          "com.google.devtools.build.lib.analysis.starlark.StarlarkTransitionTest.",
          "com.google.devtools.build.lib.analysis.util.ConfigurationTestCase.",
          "com.google.devtools.build.lib.analysis.util.DummyTestFragment.",
          "com.google.devtools.build.lib.buildtool.ConvenienceSymlinkTest.",
          "com.google.devtools.build.lib.rules.config.ConfigSettingTest.",
          "com.google.devtools.build.lib.runtime.AbstractCommandTest.",
          "com.google.devtools.build.lib.runtime.BlazeCommandDispatcherRcoptionsTest.",
          "com.google.devtools.build.lib.runtime.BlazeCommandDispatcherTest.",
          "com.google.devtools.build.lib.runtime.CommandInterruptionTest.",
          "com.google.devtools.build.lib.skyframe.config.ParsedFlagsValueTest.",
          "com.google.devtools.build.lib.skyframe.config.PlatformMappingFunctionTest.",
          "com.google.devtools.build.lib.skyframe.config.PlatformMappingValueTest.",
          "com.google.devtools.build.lib.testing.common.FakeOptionsTest.",
          "com.google.devtools.build.lib.util.OptionsUtilsTest.",
          "com.google.devtools.build.lib.worker.ExampleWorkerMultiplexerOptions",
          "com.google.devtools.build.lib.worker.ExampleWorkerOptions",
          "com.google.devtools.common.options.BoolOrEnumConverterTest.",
          "com.google.devtools.common.options.EnumConverterTest.",
          "com.google.devtools.common.options.FieldOptionDefinitionTest.",
          "com.google.devtools.common.options.OptionsDataTest.",
          "com.google.devtools.common.options.OptionsMapConversionTest.",
          "com.google.devtools.common.options.OptionsParserTest.",
          "com.google.devtools.common.options.OptionsTest.",
          "com.google.devtools.common.options.processor.OptionProcessorTest.",
          "com.google.devtools.common.options.testing.OptionsTesterTest.",
          "com.google.devtools.common.options.TestOptions");

  private void checkDeprecated(ExecutableElement method) throws OptionProcessorException {
    Option annotation = method.getAnnotation(Option.class);
    ImmutableList<OptionEffectTag> effectTags = ImmutableList.copyOf(annotation.effectTags());
    ImmutableList<OptionMetadataTag> metadataTags = ImmutableList.copyOf(annotation.metadataTags());
    boolean hasDeprecatedAnnotation = method.getAnnotation(Deprecated.class) != null;
    boolean hasDeprecatedMetadataTag = metadataTags.contains(OptionMetadataTag.DEPRECATED);

    if (effectTags.contains(OptionEffectTag.NO_OP)
        && !metadataTags.contains(OptionMetadataTag.HIDDEN)
        && !metadataTags.contains(OptionMetadataTag.INTERNAL)
        && !hasDeprecatedAnnotation) {
      // Allowlist for tests - these are in the process of being fixed.
      String enclosingClassName = method.getEnclosingElement().toString();
      boolean allowlisted =
          NO_OP_OPTION_ALLOWLIST.stream().anyMatch(enclosingClassName::startsWith);
      if (!allowlisted) {
        throw new OptionProcessorException(
            method,
            "No-op options must be annotated with @Deprecated, or have metadata tag HIDDEN or"
                + " INTERNAL. Alternatively add %s to the allowlist.",
            enclosingClassName);
      }
    }

    if (hasDeprecatedMetadataTag && !hasDeprecatedAnnotation) {
      throw new OptionProcessorException(
          method, "Options with metadata tag DEPRECATED must be annotated with @Deprecated.");
    }
    if (hasDeprecatedAnnotation && !hasDeprecatedMetadataTag) {
      throw new OptionProcessorException(
          method, "Options annotated with @Deprecated must have metadata tag DEPRECATED.");
    }
  }

  private void checkConverter(ExecutableElement method) throws OptionProcessorException {
    TypeMirror optionType = method.getReturnType();
    Option annotation = method.getAnnotation(Option.class);
    ImmutableList<TypeMirror> acceptedConverterReturnTypes =
        getAcceptedConverterReturnTypes(method);

    // For simple, static expansions, don't accept non-Void types.
    if (annotation.expansion().length != 0
        && !typeUtils.isSameType(
            optionType, elementUtils.getTypeElement(Void.class.getCanonicalName()).asType())) {
      throw new OptionProcessorException(
          method,
          "Option is an expansion flag with a static expansion, but does not have Void type.");
    }

    // Obtain the converter for this option.
    AnnotationMirror optionMirror =
        ProcessorUtils.getAnnotation(elementUtils, typeUtils, method, Option.class);
    TypeElement defaultConverterElement =
        elementUtils.getTypeElement(Converter.class.getCanonicalName());
    TypeElement converterElement =
        ProcessorUtils.getClassTypeFromAnnotationField(elementUtils, optionMirror, "converter");

    if (typeUtils.isSameType(converterElement.asType(), defaultConverterElement.asType())) {
      // Find a matching converter in the default converter list, and check that it successfully
      // parses the default value for this option.
      checkForDefaultConverter(method, acceptedConverterReturnTypes, annotation.defaultValue());
    } else {
      // Check that the provided converter has an accepted return type.
      checkProvidedConverter(method, acceptedConverterReturnTypes, converterElement);
    }
  }

  private ImmutableList<TypeMirror> getAcceptedConverterReturnTypes(ExecutableElement method)
      throws OptionProcessorException {
    TypeMirror optionType = method.getReturnType();
    Option annotation = method.getAnnotation(Option.class);
    TypeMirror listType = elementUtils.getTypeElement(List.class.getCanonicalName()).asType();

    if (annotation.allowMultiple()) {
      if (optionType.getKind() != TypeKind.DECLARED) {
        throw new OptionProcessorException(
            method,
            "Option that allows multiple occurrences must be of type %s, but is of type %s",
            listType,
            optionType);
      }
      DeclaredType optionDeclaredType = (DeclaredType) optionType;
      if (!typeUtils.isAssignable(typeUtils.erasure(optionDeclaredType), listType)) {
        throw new OptionProcessorException(
            method,
            "Option that allows multiple occurrences must be assignable to type %s, but is of type"
                + " %s",
            listType,
            optionType);
      }
      List<? extends TypeMirror> genericParameters = optionDeclaredType.getTypeArguments();
      if (genericParameters.size() != 1) {
        throw new OptionProcessorException(
            method,
            "Option that allows multiple occurrences must be of type %s, where E is the type of an"
                + " individual command-line mention of this option, but is of type %s",
            listType,
            optionType);
      }
      return ImmutableList.of(genericParameters.get(0), optionType);
    } else {
      return ImmutableList.of(optionType);
    }
  }

  private void checkForDefaultConverter(
      ExecutableElement method, List<TypeMirror> acceptedConverterReturnTypes, String defaultValue)
      throws OptionProcessorException {
    if (defaultConverters == null) {
      // Bootstrapping. Do not do this check.
      return;
    }

    for (TypeMirror acceptedConverterReturnType : acceptedConverterReturnTypes) {
      Converter<?> converterInstance = findDefaultConverter(acceptedConverterReturnType);
      if (converterInstance == null) {
        continue;
      }
      try {
        converterInstance.convert(defaultValue, null);
      } catch (OptionsParsingException e) {
        TypeElement converter =
            elementUtils.getTypeElement(converterInstance.getClass().getCanonicalName());
        throw new OptionProcessorException(
            method,
            e,
            "Option lists a default value (%s) that is not parsable by the option's converter (%s)",
            defaultValue,
            converter);
      }
      return;
    }
    throw new OptionProcessorException(
        method,
        "Cannot find valid converter for option of type %s",
        acceptedConverterReturnTypes.get(0));
  }

  @Nullable
  private Converter<?> findDefaultConverter(TypeMirror type) {
    // According to the documentation of TypeMirror, equality check is not how one checks whether
    // two instances reference the same type but Types.isSameType().
    for (Map.Entry<TypeMirror, Converter<?>> entry : defaultConverters.entrySet()) {
      if (typeUtils.isSameType(type, entry.getKey())) {
        return entry.getValue();
      }
    }
    return null;
  }

  private void checkProvidedConverter(
      ExecutableElement method,
      ImmutableList<TypeMirror> acceptedConverterReturnTypes,
      TypeElement converterElement)
      throws OptionProcessorException {
    if (converterElement.getModifiers().contains(Modifier.ABSTRACT)) {
      throw new OptionProcessorException(
          method, "The converter type %s must be a concrete type", converterElement.asType());
    }

    DeclaredType converterType = (DeclaredType) converterElement.asType();
    List<ExecutableElement> methodList =
        elementUtils.getAllMembers(converterElement).stream()
            .filter(element -> element.getKind() == ElementKind.METHOD)
            .map(methodElement -> (ExecutableElement) methodElement)
            .filter(methodElement -> methodElement.getSimpleName().contentEquals("convert"))
            .filter(
                methodElement ->
                    methodElement.getParameters().size() == 2
                        && typeUtils.isSameType(
                            methodElement.getParameters().get(0).asType(),
                            elementUtils.getTypeElement(String.class.getCanonicalName()).asType())
                        && typeUtils.isSameType(
                            methodElement.getParameters().get(1).asType(),
                            elementUtils.getTypeElement(Object.class.getCanonicalName()).asType()))
            .collect(Collectors.toList());

    if (methodList.size() != 1) {
      throw new OptionProcessorException(
          method,
          "Converter %s has %d methods 'convert(String, Object)', expected 1: %s",
          converterElement,
          methodList.size(),
          methodList.stream().map(Object::toString).collect(Collectors.joining(", ")));
    }

    ExecutableType convertMethodType =
        (ExecutableType) typeUtils.asMemberOf(converterType, methodList.get(0));
    TypeMirror convertMethodResultType = convertMethodType.getReturnType();
    for (TypeMirror acceptedConverterReturnType : acceptedConverterReturnTypes) {
      if (typeUtils.isAssignable(convertMethodResultType, acceptedConverterReturnType)) {
        return;
      }
    }
    throw new OptionProcessorException(
        method,
        "Type of field (%s) must be assignable from the converter's return type (%s)",
        acceptedConverterReturnTypes.get(0),
        convertMethodResultType);
  }
}
