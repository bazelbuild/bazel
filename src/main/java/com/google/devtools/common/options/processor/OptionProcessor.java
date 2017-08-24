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
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedSourceVersion;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.Diagnostic;

/**
 * Annotation processor for {@link Option}.
 *
 * <p>The {@link OptionsParser} only accepts publicly declared options in {@link
 * OptionsBase}-inheriting classes, and there is no support for {@link Option} annotated fields
 * declared elsewhere or privately. Prevent such uses from compiling.
 */
@SupportedAnnotationTypes({"com.google.devtools.common.options.Option"})
@SupportedSourceVersion(SourceVersion.RELEASE_8)
public final class OptionProcessor extends AbstractProcessor {

  private Types typeUtils;
  private Elements elementUtils;
  private Messager messager;

  @Override
  public synchronized void init(ProcessingEnvironment processingEnv) {
    super.init(processingEnv);
    typeUtils = processingEnv.getTypeUtils();
    elementUtils = processingEnv.getElementUtils();
    messager = processingEnv.getMessager();
  }

  private static class OptionProcessorException extends Exception {
    private Element elementInError;

    OptionProcessorException(Element element, String message, Object... args) {
      super(String.format(message, args));
      elementInError = element;
    }
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

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    for (Element annotatedElement : roundEnv.getElementsAnnotatedWith(Option.class)) {
      try {
        // Only fields are annotated with Option, this should already be checked by the
        // @Target(ElementType.FIELD) annotation.
        VariableElement optionField = (VariableElement) annotatedElement;

        checkModifiers(optionField);
        checkInOptionBase(optionField);
        checkEffectTagRationality(optionField);
        checkMetadataTagAndCategoryRationality(optionField);
      } catch (OptionProcessorException e) {
        error(e.elementInError, e.getMessage());
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
