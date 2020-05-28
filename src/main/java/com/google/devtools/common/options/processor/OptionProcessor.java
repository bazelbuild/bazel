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
package com.google.devtools.common.options.processor;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.ExpansionFunction;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
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
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.ExecutableType;
import javax.lang.model.type.PrimitiveType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.Diagnostic;

/**
 * Annotation processor for {@link Option}.
 *
 * <p>Checks the following invariants about {@link Option}-annotated fields ("options"):
 *
 * <ul>
 *   <li>The {@link OptionsParser} only accepts options in {@link OptionsBase}-inheriting classes
 *   <li>All options must be declared publicly and be neither static nor final.
 *   <li>All options that must be used on the command line must have sensible names without
 *       whitespace or other confusing characters, such as equal signs.
 *   <li>The type of the option must match the converter that will convert the unparsed string value
 *       into the option type. For options that do not specify a converter, check that there is a
 *       valid match in the {@link Converters#DEFAULT_CONVERTERS} list.
 *   <li>Options must list valid combinations of tags and documentation categories.
 *   <li>Expansion options and options with implicit requirements cannot expand in more than one
 *       way, how multiple expansions would interact is not defined and should not be necessary.
 *   <li>Multiple options must not declare default value (see {@link
 *       #MULTIPLE_OPTIONS_DEFAULT_VALUE_EXCEPTIONS} for exceptions).
 * </ul>
 *
 * <p>These properties can be relied upon at runtime without additional checks.
 */
@SupportedAnnotationTypes({"com.google.devtools.common.options.Option"})
public final class OptionProcessor extends AbstractProcessor {

  private Types typeUtils;
  private Elements elementUtils;
  private Messager messager;
  private ImmutableMap<TypeMirror, Converter<?>> defaultConverters;
  private ImmutableMap<Class<?>, PrimitiveType> primitiveTypeMap;

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported();
  }

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    typeUtils = processingEnv.getTypeUtils();
    elementUtils = processingEnv.getElementUtils();
    messager = processingEnv.getMessager();

    // Because of the discrepancies between the java.lang and javax.lang type models, we can't
    // directly use the get() method for the default converter map. Instead, we'll convert it once,
    // to be more usable, and with the boxed type return values of convert() as the keys.
    ImmutableMap.Builder<TypeMirror, Converter<?>> converterMapBuilder =
        new ImmutableMap.Builder<>();

    // Create a link from the primitive Classes to their primitive types. This intentionally
    // only contains the types in the DEFAULT_CONVERTERS map.
    ImmutableMap.Builder<Class<?>, PrimitiveType> builder = new ImmutableMap.Builder<>();
    builder.put(int.class, typeUtils.getPrimitiveType(TypeKind.INT));
    builder.put(double.class, typeUtils.getPrimitiveType(TypeKind.DOUBLE));
    builder.put(boolean.class, typeUtils.getPrimitiveType(TypeKind.BOOLEAN));
    builder.put(long.class, typeUtils.getPrimitiveType(TypeKind.LONG));
    primitiveTypeMap = builder.build();

    for (Map.Entry<Class<?>, Converter<?>> entry : Converters.DEFAULT_CONVERTERS.entrySet()) {
      Class<?> converterClass = entry.getKey();
      String typeName = converterClass.getCanonicalName();
      TypeElement typeElement = elementUtils.getTypeElement(typeName);
      // Check that we can get a type mirror, either through the type element or the primitive type.
      if (typeElement != null) {
        converterMapBuilder.put(typeElement.asType(), entry.getValue());
      } else {
        if (!primitiveTypeMap.containsKey(converterClass)) {
          messager.printMessage(
              Diagnostic.Kind.ERROR,
              String.format("Can't get a TypeElement for Type %s", typeName));
          continue;
        }
        // Add the primitive types to the map, both in primitive TypeMirror form, and the boxed
        // classes, such as java.lang.Integer, because primitives must be boxed in collections,
        // such as allowMultiple options, which have type List<singleOptionType>.
        PrimitiveType primitiveType = primitiveTypeMap.get(converterClass);
        converterMapBuilder.put(primitiveType, entry.getValue());
        converterMapBuilder.put(typeUtils.boxedClass(primitiveType).asType(), entry.getValue());
      }
    }
    defaultConverters = converterMapBuilder.build();
  }

  /** Check that the Option variables only occur in OptionBase-inheriting classes. */
  private void checkInOptionBase(VariableElement optionField) throws OptionProcessorException {
    if (optionField.getEnclosingElement().getKind() != ElementKind.CLASS) {
      throw new OptionProcessorException(optionField, "The field should belong to a class.");
    }
    TypeMirror thisOptionClass = optionField.getEnclosingElement().asType();
    TypeMirror optionsBase =
        elementUtils.getTypeElement("com.google.devtools.common.options.OptionsBase").asType();
    if (!typeUtils.isAssignable(thisOptionClass, optionsBase)) {
      throw new OptionProcessorException(
          optionField,
          "@Option annotated fields can only be in classes that inherit from OptionsBase.");
    }
  }

  /**
   * Checks that the Option variables is public and neither final nor static.
   *
   * <p>Private or protected fields would prevent the options parser from having full access to the
   * fields it's expected to read, and {@link OptionsBase} equality would not work as intended.
   *
   * <p>Static or final fields would cause issue with correct value assigning at the end of parsing.
   */
  private void checkModifiers(VariableElement optionField) throws OptionProcessorException {
    if (!optionField.getModifiers().contains(Modifier.PUBLIC)) {
      throw new OptionProcessorException(optionField, "@Option annotated fields should be public.");
    }
    if (optionField.getModifiers().contains(Modifier.STATIC)) {
      throw new OptionProcessorException(
          optionField, "@Option annotated fields should not be static.");
    }
    if (optionField.getModifiers().contains(Modifier.FINAL)) {
      throw new OptionProcessorException(
          optionField, "@Option annotated fields should not be final.");
    }
  }

  private ImmutableList<TypeMirror> getAcceptedConverterReturnTypes(VariableElement optionField)
      throws OptionProcessorException {
    TypeMirror optionType = optionField.asType();
    Option annotation = optionField.getAnnotation(Option.class);
    TypeMirror listType = elementUtils.getTypeElement(List.class.getCanonicalName()).asType();
    // Options that accumulate multiple mentions in an arglist must have type List<T>, where each
    // individual mention has type T. Identify type T to use it for checking the converter's return
    // type.
    if (annotation.allowMultiple()) {
      // Check that the option type is in fact a list.
      if (optionType.getKind() != TypeKind.DECLARED) {
        throw new OptionProcessorException(
            optionField,
            "Option that allows multiple occurrences must be of type %s, but is of type %s",
            listType,
            optionType);
      }
      DeclaredType optionDeclaredType = (DeclaredType) optionType;
      // optionDeclaredType.asElement().asType() gets us from List<actualType> to List<E>, so this
      // is unfortunately necessary.
      if (!typeUtils.isAssignable(optionDeclaredType.asElement().asType(), listType)) {
        throw new OptionProcessorException(
            optionField,
            "Option that allows multiple occurrences must be of type %s, but is of type %s",
            listType,
            optionType);
      }

      // Check that there is only one generic parameter, and store it as the singular option type.
      List<? extends TypeMirror> genericParameters = optionDeclaredType.getTypeArguments();
      if (genericParameters.size() != 1) {
        throw new OptionProcessorException(
            optionField,
            "Option that allows multiple occurrences must be of type %s, "
                + "where E is the type of an individual command-line mention of this option, "
                + "but is of type %s",
            listType,
            optionType);
      }

      // For repeated options, we also accept cases where each option itself contains a list, which
      // are then concatenated into the final single list type. For this reason, we will accept both
      // converters that return the type of a single option, and List<singleOption>, which,
      // incidentally, is the original optionType.
      // Example: --foo=a,b,c --foo=d,e,f could have a final value of type List<Char>,
      //       value {a,b,c,e,d,f}, instead of requiring a final value of type List<List<Char>>
      //       value {{a,b,c},{d,e,f}}
      TypeMirror singularOptionType = genericParameters.get(0);

      return ImmutableList.of(singularOptionType, optionType);
    } else {
      return ImmutableList.of(optionField.asType());
    }
  }

  private void checkForDefaultConverter(
      VariableElement optionField,
      List<TypeMirror> acceptedConverterReturnTypes,
      String defaultValue)
      throws OptionProcessorException {
    for (TypeMirror acceptedConverterReturnType : acceptedConverterReturnTypes) {
      Converter<?> converterInstance = defaultConverters.get(acceptedConverterReturnType);
      if (converterInstance == null) {
        // This return type isn't a match, move on to the next one in case.
        continue;
      }
      TypeElement converter =
          elementUtils.getTypeElement(converterInstance.getClass().getCanonicalName());
      try {
        // For the default converters, it so happens we have access to the convert methods
        // at compile time, since we already have the OptionsParser source. Take advantage of
        // this to test that the provided defaultValue is valid.
        converterInstance.convert(defaultValue);
      } catch (OptionsParsingException e) {
        throw new OptionProcessorException(
            optionField,
            /* throwable = */ e,
            "Option lists a default value (%s) that is not parsable by the option's converter "
                + "(s)",
            defaultValue,
            converter);
      }
      return; // This one passes the test.
    }

    // We didn't find a default converter.
    throw new OptionProcessorException(
        optionField,
        "Cannot find valid converter for option of type %s",
        acceptedConverterReturnTypes.get(0));
  }

  private void checkProvidedConverter(
      VariableElement optionField,
      ImmutableList<TypeMirror> acceptedConverterReturnTypes,
      TypeElement converterElement)
      throws OptionProcessorException {
    if (converterElement.getModifiers().contains(Modifier.ABSTRACT)) {
      throw new OptionProcessorException(
          optionField, "The converter type %s must be a concrete type", converterElement.asType());
    }

    DeclaredType converterType = (DeclaredType) converterElement.asType();

    // Unfortunately, for provided classes, we do not have access to the compiled convert
    // method at this time, and cannot check that the default value is parseable. We will
    // instead check that T of Converter<T> matches the option's type, but this is all we can
    // do.
    List<ExecutableElement> methodList =
        elementUtils.getAllMembers(converterElement).stream()
            .filter(element -> element.getKind() == ElementKind.METHOD)
            .map(methodElement -> (ExecutableElement) methodElement)
            .filter(methodElement -> methodElement.getSimpleName().contentEquals("convert"))
            .filter(
                methodElement ->
                    methodElement.getParameters().size() == 1
                        && typeUtils.isSameType(
                            methodElement.getParameters().get(0).asType(),
                            elementUtils.getTypeElement(String.class.getCanonicalName()).asType()))
            .collect(Collectors.toList());
    // Check that there is just the one method
    if (methodList.size() != 1) {
      throw new OptionProcessorException(
          optionField,
          "Converter %s has methods 'convert(String)': %s",
          converterElement,
          methodList.stream().map(Object::toString).collect(Collectors.joining(", ")));
    }

    ExecutableType convertMethodType =
        (ExecutableType) typeUtils.asMemberOf(converterType, methodList.get(0));
    TypeMirror convertMethodResultType = convertMethodType.getReturnType();
    // Check that the converter's return type is in the accepted list.
    for (TypeMirror acceptedConverterReturnType : acceptedConverterReturnTypes) {
      if (typeUtils.isAssignable(convertMethodResultType, acceptedConverterReturnType)) {
        return; // This one passes the test.
      }
    }
    throw new OptionProcessorException(
        optionField,
        "Type of field (%s) must be assignable from the converter's return type (%s)",
        acceptedConverterReturnTypes.get(0),
        convertMethodResultType);
  }

  private void checkConverter(VariableElement optionField) throws OptionProcessorException {
    TypeMirror optionType = optionField.asType();
    Option annotation = optionField.getAnnotation(Option.class);
    ImmutableList<TypeMirror> acceptedConverterReturnTypes =
        getAcceptedConverterReturnTypes(optionField);

    // For simple, static expansions, don't accept non-Void types.
    if (annotation.expansion().length != 0
        && !typeUtils.isSameType(
            optionType, elementUtils.getTypeElement(Void.class.getCanonicalName()).asType())) {
      throw new OptionProcessorException(
          optionField,
          "Option is an expansion flag with a static expansion, but does not have Void type.");
    }

    // Obtain the converter for this option.
    AnnotationMirror optionMirror =
        ProcessorUtils.getAnnotation(elementUtils, typeUtils, optionField, Option.class);
    TypeElement defaultConverterElement =
        elementUtils.getTypeElement(Converter.class.getCanonicalName());
    TypeElement converterElement =
        ProcessorUtils.getClassTypeFromAnnotationField(elementUtils, optionMirror, "converter");
    if (converterElement == null) {
      throw new OptionProcessorException(optionField, "Null converter found.");
    }

    if (typeUtils.isSameType(converterElement.asType(), defaultConverterElement.asType())) {
      // Find a matching converter in the default converter list, and check that it successfully
      // parses the default value for this option.
      checkForDefaultConverter(
          optionField, acceptedConverterReturnTypes, annotation.defaultValue());
    } else {
      // Check that the provided converter has an accepted return type.
      checkProvidedConverter(optionField, acceptedConverterReturnTypes, converterElement);
    }
  }

  /**
   * Check that the option lists at least one effect, and that no nonsensical combinations are
   * listed, such as having a known effect listed with UNKNOWN.
   */
  private void checkEffectTagRationality(VariableElement optionField)
      throws OptionProcessorException {
    Option annotation = optionField.getAnnotation(Option.class);
    OptionEffectTag[] effectTags = annotation.effectTags();
    // Check that there is at least one OptionEffectTag listed.
    if (effectTags.length < 1) {
      throw new OptionProcessorException(
          optionField,
          "Option does not list at least one OptionEffectTag. If the option has no effect, "
              + "please be explicit and add NO_OP. Otherwise, add a tag representing its effect.");
    } else if (effectTags.length > 1) {
      // If there are more than 1 tag, make sure that NO_OP and UNKNOWN is not one of them.
      // These don't make sense if other effects are listed.
      ImmutableList<OptionEffectTag> tags = ImmutableList.copyOf(effectTags);
      if (tags.contains(OptionEffectTag.UNKNOWN)) {
        throw new OptionProcessorException(
            optionField,
            "Option includes UNKNOWN with other, known, effects. Please remove UNKNOWN from "
                + "the list.");
      }
      if (tags.contains(OptionEffectTag.NO_OP)) {
        throw new OptionProcessorException(
            optionField,
            "Option includes NO_OP with other effects. This doesn't make much sense. Please "
                + "remove NO_OP or the actual effects from the list, whichever is correct.");
      }
    }
  }

  /**
   * Check that if the metadata tags listed by an option require the option to be unknown by the
   * average user, the same option will be omitted from documentation.
   */
  private void checkMetadataTagAndCategoryRationality(VariableElement optionField)
      throws OptionProcessorException {
    Option annotation = optionField.getAnnotation(Option.class);
    OptionMetadataTag[] metadataTags = annotation.metadataTags();
    OptionDocumentationCategory category = annotation.documentationCategory();

    for (OptionMetadataTag tag : metadataTags) {
      if (tag == OptionMetadataTag.HIDDEN || tag == OptionMetadataTag.INTERNAL) {
        if (category != OptionDocumentationCategory.UNDOCUMENTED) {
          throw new OptionProcessorException(
              optionField,
              "Option has metadata tag %s but does not have category UNDOCUMENTED. Please fix.",
              tag);
        }
      }
    }
  }

  /** These categories used to indicate whether a flag was documented, but no longer. */
  private static final ImmutableList<String> DEPRECATED_CATEGORIES =
      ImmutableList.of("undocumented", "hidden", "internal");

  private void checkOldCategoriesAreNotUsed(VariableElement optionField)
      throws OptionProcessorException {
    Option annotation = optionField.getAnnotation(Option.class);
    if (DEPRECATED_CATEGORIES.contains(annotation.category())) {
      throw new OptionProcessorException(
          optionField,
          "Documentation level is no longer read from the option category. Category \""
              + annotation.category()
              + "\" is disallowed, see OptionMetadataTags for the relevant tags.");
    }
  }

  private void checkOptionName(VariableElement optionField) throws OptionProcessorException {
    Option annotation = optionField.getAnnotation(Option.class);
    String optionName = annotation.name();
    if (optionName.isEmpty()) {
      throw new OptionProcessorException(optionField, "Option must have an actual name.");
    }

    // Specifically for non-internal options, which are flags intended to be used on the command
    // line, check that there are no weird characters or whitespace.
    if (!ImmutableList.copyOf(annotation.metadataTags()).contains(OptionMetadataTag.INTERNAL)) {
      if (!Pattern.matches("([\\w:-])*", optionName)) {
        // Ideally, this would be just \w, but - and : are needed for legacy options. We can lie in
        // the error though, no harm in encouraging good behavior.
        throw new OptionProcessorException(
            optionField,
            "Options that are used on the command line as flags must have names made from word "
                + "characters only.");
      }
    }
  }

  /**
   * Some flags expand to other flags, either in place, or with "implicit requirements" that get
   * added on top of the flag's value. Don't let these flags do too many crazy things, dealing with
   * this is enough.
   */
  private void checkExpansionOptions(VariableElement optionField) throws OptionProcessorException {
    Option annotation = optionField.getAnnotation(Option.class);
    boolean isStaticExpansion = annotation.expansion().length > 0;
    boolean hasImplicitRequirements = annotation.implicitRequirements().length > 0;

    AnnotationMirror annotationMirror =
        ProcessorUtils.getAnnotation(elementUtils, typeUtils, optionField, Option.class);
    TypeElement expansionFunction =
        ProcessorUtils.getClassTypeFromAnnotationField(
            elementUtils, annotationMirror, "expansionFunction");
    TypeElement defaultExpansionFunction =
        elementUtils.getTypeElement(ExpansionFunction.class.getCanonicalName());
    boolean isFunctionalExpansion =
        !typeUtils.isSameType(expansionFunction.asType(), defaultExpansionFunction.asType());

    if (isStaticExpansion && isFunctionalExpansion) {
      throw new OptionProcessorException(
          optionField,
          "Options cannot expand using both a static expansion list and an expansion function.");
    }
    boolean isExpansion = isStaticExpansion || isFunctionalExpansion;

    if (isExpansion && hasImplicitRequirements) {
      throw new OptionProcessorException(
          optionField,
          "Can't set an option to be both an expansion option and have implicit requirements.");
    }

    if (isExpansion || hasImplicitRequirements) {
      if (annotation.allowMultiple()) {
        throw new OptionProcessorException(
            optionField,
            "Can't set an option to accumulate multiple values and let it expand to other flags.");
      }
    }
  }

  private boolean hasSpecialNullDefaultValue(Option annotation) {
    return OptionDefinition.SPECIAL_NULL_DEFAULT_VALUE.equals(annotation.defaultValue());
  }

  /**
   * Options that are allowed to have default values.
   *
   * <p>DO NOT ADD new (especially production) options here - the long-term goal is to prohibit
   * multiple options to have default values.
   */
  private static final ImmutableList<String> MULTIPLE_OPTIONS_DEFAULT_VALUE_EXCEPTIONS =
      ImmutableList.of(
          // Multiple options used in OptionDefinitionTest
          "non_empty_string_multiple_option",
          "empty_string_multiple_option",
          // Production multiple options that still have default value.
          // Mostly due to backward compatibility reasons.
          "runs_per_test",
          "flaky_test_attempts",
          "worker_max_instances");

  private boolean isMultipleOptionDefaultValueException(Option annotation) {
    return MULTIPLE_OPTIONS_DEFAULT_VALUE_EXCEPTIONS.contains(annotation.name());
  }

  private void checkNoDefaultValueForMultipleOption(VariableElement optionField)
      throws OptionProcessorException {
    Option annotation = optionField.getAnnotation(Option.class);

    if (annotation.allowMultiple()
        && !hasSpecialNullDefaultValue(annotation)
        && !isMultipleOptionDefaultValueException(annotation)) {
      String message =
          String.format(
              "Default values for multiple options are not allowed - use \"%s\" special value",
              "null");
      throw new OptionProcessorException(optionField, message);
    }
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    for (Element annotatedElement : roundEnv.getElementsAnnotatedWith(Option.class)) {
      try {
        // Only fields are annotated with Option, this should already be checked by the
        // @Target(ElementType.FIELD) annotation.
        VariableElement optionField = (VariableElement) annotatedElement;

        checkModifiers(optionField);
        checkInOptionBase(optionField);
        checkOptionName(optionField);
        checkOldCategoriesAreNotUsed(optionField);
        checkExpansionOptions(optionField);
        checkConverter(optionField);
        checkEffectTagRationality(optionField);
        checkMetadataTagAndCategoryRationality(optionField);
        checkNoDefaultValueForMultipleOption(optionField);
      } catch (OptionProcessorException e) {
        error(e.getElementInError(), e.getMessage());
      }
    }
    // Claim all Option annotated fields.
    return true;
  }

  /**
   * Prints an error message & fails the compilation.
   *
   * @param e The element which has caused the error. Can be null
   * @param msg The error message
   */
  public void error(Element e, String msg) {
    messager.printMessage(Diagnostic.Kind.ERROR, msg, e);
  }
}
