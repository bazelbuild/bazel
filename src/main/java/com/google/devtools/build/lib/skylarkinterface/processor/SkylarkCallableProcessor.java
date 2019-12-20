// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkinterface.processor;

import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.errorprone.annotations.FormatMethod;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
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
import javax.lang.model.type.WildcardType;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.Diagnostic;

/**
 * Annotation processor for {@link SkylarkCallable}.
 *
 * <p>Checks the following invariants about {@link SkylarkCallable}-annotated methods:
 *
 * <ul>
 *   <li>The method must be public.
 *   <li>The method must be non-static.
 *   <li>If structField=true, there must be zero user-supplied parameters.
 *   <li>Method parameters must be supplied in the following order:
 *       <pre>method([positionals]*[other user-args](Location)(StarlarkThread)(StarlarkSemantics))
 *       </pre>
 *       where Location, StarlarkThread, and StarlarkSemantics are supplied by the interpreter if
 *       and only if useLocation, useStarlarkThread, or useStarlarkSemantics options are specified,
 *       respectively.
 *   <li>The number of method parameters must match the number of annotation-declared parameters
 *       plus the number of interpreter-supplied parameters.
 *   <li>Each parameter, if explicitly typed, may only use either 'type' or 'allowedTypes', not
 *       both.
 *   <li>Parameters may not specify their generic types (they must use the <code>?</code> wildcard
 *       exclusively.
 *   <li>Noneable parameters must have Java parameter type Object, as the actual value may be either
 *       {@code None} or some other value, which do not share a superclass other than Object (or
 *       StarlarkValue, which is typically no more descriptive than Object).
 *   <li>Each parameter must be positional or named (or both).
 *   <li>Positional-only parameters must be specified before any named parameters.
 *   <li>Positional parameters must be specified before any non-positional parameters.
 *   <li>Positional parameters without default values must be specified before any positional
 *       parameters with default values.
 *   <li>Either the doc string is non-empty, or documented is false.
 *   <li>Each class may only have one annotated method with selfCall=true.
 *   <li>A method annotated with selfCall=true must have a non-empty name.
 *   <li>A method annotated with selfCall=true must have structField=false.
 *   <li>The method's class must implement StarlarkValue.
 *   <li>The class of the declared result type, if final, must be accepted by {@link
 *       Starlark#fromJava}.
 * </ul>
 *
 * <p>These properties can be relied upon at runtime without additional checks.
 */
@SupportedAnnotationTypes({
  "com.google.devtools.build.lib.skylarkinterface.SkylarkCallable",
  "com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary",
  "com.google.devtools.build.lib.skylarkinterface.SkylarkModule"
})
public final class SkylarkCallableProcessor extends AbstractProcessor {
  private Messager messager;

  // A set containing the names of all classes which have a method with @SkylarkCallable.selfCall.
  private Set<String> classesWithSelfcall;
  // A multimap where keys are class names, and values are the callable method names identified in
  // that class (where "method name" is @SkylarkCallable.name").
  private SetMultimap<String, String> processedClassMethods;

  private Types types;
  private Elements elements;

  private static final String SKYLARK_LIST = "com.google.devtools.build.lib.syntax.Sequence<?>";
  private static final String SKYLARK_DICT = "com.google.devtools.build.lib.syntax.Dict<?,?>";
  private static final String LOCATION = "com.google.devtools.build.lib.events.Location";
  private static final String STARLARK_THREAD =
      "com.google.devtools.build.lib.syntax.StarlarkThread";
  private static final String STARLARK_SEMANTICS =
      "com.google.devtools.build.lib.syntax.StarlarkSemantics";

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported();
  }

  @Override
  public synchronized void init(ProcessingEnvironment env) {
    super.init(env);
    this.types = env.getTypeUtils();
    this.elements = env.getElementUtils();
    messager = env.getMessager();
    classesWithSelfcall = new HashSet<>();
    processedClassMethods = LinkedHashMultimap.create();
  }

  private TypeMirror getType(String canonicalName) {
    return elements.getTypeElement(canonicalName).asType();
  }

  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    TypeMirror stringType = getType("java.lang.String");
    TypeMirror integerType = getType("java.lang.Integer");
    TypeMirror booleanType = getType("java.lang.Boolean");
    TypeMirror listType = getType("java.util.List");
    TypeMirror mapType = getType("java.util.Map");
    TypeMirror skylarkValueType = getType("com.google.devtools.build.lib.syntax.StarlarkValue");

    // Ensure SkylarkModule-annotated classes implement StarlarkValue.
    for (Element cls : roundEnv.getElementsAnnotatedWith(SkylarkModule.class)) {
      if (!types.isAssignable(cls.asType(), skylarkValueType)) {
        errorf(
            cls,
            "class %s has @SkylarkModule annotation but does not implement StarlarkValue",
            cls.getSimpleName());
      }
    }

    // TODO(adonovan): reject a SkylarkCallable-annotated method whose class doesn't have (or
    // inherit) a SkylarkModule documentation annotation.

    // Only SkylarkGlobalLibrary-annotated classes, and those that implement StarlarkValue,
    // are allowed SkylarkCallable-annotated methods.
    Set<Element> okClasses =
        new HashSet<>(roundEnv.getElementsAnnotatedWith(SkylarkGlobalLibrary.class));

    for (Element element : roundEnv.getElementsAnnotatedWith(SkylarkCallable.class)) {
      // Only methods are annotated with SkylarkCallable. This is verified by the
      // @Target(ElementType.METHOD) annotation.
      ExecutableElement methodElement = (ExecutableElement) element;
      SkylarkCallable annotation = methodElement.getAnnotation(SkylarkCallable.class);

      if (!methodElement.getModifiers().contains(Modifier.PUBLIC)) {
        error(methodElement, "@SkylarkCallable annotated methods must be public.");
      }
      if (methodElement.getModifiers().contains(Modifier.STATIC)) {
        error(methodElement, "@SkylarkCallable annotated methods cannot be static.");
      }

      try {
        verifyNameNotEmpty(methodElement, annotation);
        verifyDocumented(methodElement, annotation);
        verifyNotStructFieldWithParams(methodElement, annotation);
        verifyParameters(methodElement, annotation);
        verifyExtraInterpreterParams(methodElement, annotation);
        verifyIfSelfCall(methodElement, annotation);
        verifyFlagToggles(methodElement, annotation);
        verifyNoNameConflict(methodElement, annotation);
      } catch (SkylarkCallableProcessorException exception) {
        // TODO(adonovan): don't use exceptions; report multiple errors per pass
        // as this saves time in compiler-driven refactoring.
        // This also allows us to report multiple locations,
        // such as a Param annotation and the parameter.
        error(exception.element, exception.errorMessage);
      }

      // Verify that result type, if final, might satisfy Starlark.fromJava.
      // (If the type is non-final we can't prove that all subclasses are invalid.)
      TypeMirror ret = methodElement.getReturnType();
      if (ret.getKind() == TypeKind.DECLARED) {
        DeclaredType obj = (DeclaredType) ret;
        if (obj.asElement().getModifiers().contains(Modifier.FINAL)
            && !types.isSameType(ret, stringType)
            && !types.isSameType(ret, integerType)
            && !types.isSameType(ret, booleanType)
            && !types.isAssignable(obj, skylarkValueType)
            && !types.isAssignable(obj, listType)
            && !types.isAssignable(obj, mapType)) {
          errorf(
              methodElement,
              "@SkylarkCallable-annotated method %s returns %s, which has no legal Starlark values"
                  + " (see Starlark.fromJava)",
              methodElement.getSimpleName(),
              ret);
        }
      }

      // Check that the method's class is SkylarkGlobalLibrary-annotated,
      // or implements StarlarkValue, or an error has already been reported.
      Element cls = methodElement.getEnclosingElement();
      if (okClasses.add(cls) && !types.isAssignable(cls.asType(), skylarkValueType)) {
        errorf(
            cls,
            "method %s has @SkylarkCallable annotation but enclosing class %s does not implement"
                + " StarlarkValue nor has @SkylarkGlobalLibrary annotation",
            methodElement.getSimpleName(),
            cls.getSimpleName());
      }
    }

    // Returning false allows downstream processors to work on the same annotations
    return false;
  }

  private void verifyNoNameConflict(ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {
    boolean methodNameIsUniqueForClass =
        processedClassMethods.put(
            methodElement.getEnclosingElement().asType().toString(),
            annotation.name());
    if (!methodNameIsUniqueForClass) {
      throw new SkylarkCallableProcessorException(
          methodElement,
          String.format("Containing class has more than one method with name '%s' defined.",
              annotation.name()));
    }
  }

  private void verifyFlagToggles(ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {
    if (annotation.enableOnlyWithFlag() != FlagIdentifier.NONE
        && annotation.disableWithFlag() != FlagIdentifier.NONE) {
      throw new SkylarkCallableProcessorException(
          methodElement,
          "Only one of @SkylarkCallable.enablingFlag and @SkylarkCallable.disablingFlag may be "
              + "specified.");
    }
  }

  private void verifyNameNotEmpty(ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {
    if (annotation.name().isEmpty()) {
      throw new SkylarkCallableProcessorException(
          methodElement,
          "@SkylarkCallable.name must be non-empty.");
    }
  }

  private void verifyIfSelfCall(ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {
    if (annotation.selfCall()) {
      if (annotation.structField()) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            "@SkylarkCallable-annotated methods with selfCall=true must have structField=false");
      }
      if (!classesWithSelfcall.add(methodElement.getEnclosingElement().asType().toString())) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            "Containing class has more than one selfCall method defined.");
      }
    }
  }

  private void verifyDocumented(ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {
    if (annotation.documented() && annotation.doc().isEmpty()) {
      throw new SkylarkCallableProcessorException(
          methodElement,
          "The 'doc' string must be non-empty if 'documented' is true.");
    }
  }

  private void verifyNotStructFieldWithParams(
      ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {
    if (annotation.structField()) {
      if (annotation.useStarlarkThread()
          || !annotation.extraPositionals().name().isEmpty()
          || !annotation.extraKeywords().name().isEmpty()) {
        // TODO(adonovan): decide on the restrictions.
        // - useLocation is needed only by repository_ctx.os. Abolish?
        // - useStarlarkSemantics is needed only by getSkylarkLibrariesToLink.
        // - banning useStarlarkThread has not been a problem so far,
        //   and avoids many tricky problems (especially in StructImpl.equal),
        //   but it forces implementations to assume Mutability=null,
        //   which is not quite right.
        throw new SkylarkCallableProcessorException(
            methodElement,
            "@SkylarkCallable-annotated methods with structField=true may not also specify "
                + "useStarlarkThread, extraPositionals, or extraKeywords");
      }
    }
  }

  private void verifyParameters(ExecutableElement method, SkylarkCallable annotation) {
    int numParams = method.getParameters().size();
    int numExtraInterpreterParams = numExpectedExtraInterpreterParams(annotation);
    int numParamAnnots = annotation.parameters().length;
    if (numParams != numParamAnnots + numExtraInterpreterParams) {
      errorf(
          method,
          "@SkylarkCallable annotated method has %d parameters, but annotation declared "
              + "%d user-supplied parameters and %d extra interpreter parameters.",
          numParams,
          numParamAnnots,
          numExtraInterpreterParams);
    }

    if (annotation.structField() && numParamAnnots > 0) {
      errorf(
          method,
          "@SkylarkCallable annotated methods with structField=true must have "
              + "0 user-supplied parameters. Expected %d extra interpreter parameters, "
              + "but found %d total parameters.",
          numExtraInterpreterParams,
          numParams);
    }

    TypeMirror objectType = getType("java.lang.Object");

    boolean allowPositionalNext = true;
    boolean allowPositionalOnlyNext = true;
    boolean allowNonDefaultPositionalNext = true;

    for (int i = 0; i < numParams && i < numParamAnnots; i++) {
      VariableElement param = method.getParameters().get(i);
      Param paramAnnot = annotation.parameters()[i];

      verifyParameter(param, paramAnnot, objectType);

      // Check parameter ordering.
      if (paramAnnot.positional()) {
        if (!allowPositionalNext) {
          errorf(
              param,
              "Positional parameter '%s' is specified after one or more non-positonal parameters",
              paramAnnot.name());
        }
        if (!isParamNamed(paramAnnot) && !allowPositionalOnlyNext) {
          errorf(
              param,
              "Positional-only parameter '%s' is specified after one or more named parameters",
              paramAnnot.name());
        }
        if (paramAnnot.defaultValue().isEmpty()) { // There is no default value.
          if (!allowNonDefaultPositionalNext) {
            errorf(
                param,
                "Positional parameter '%s' has no default value but is specified after one "
                    + "or more positional parameters with default values",
                paramAnnot.name());
          }
        } else { // There is a default value.
          // No positional parameters without a default value can come after this parameter.
          allowNonDefaultPositionalNext = false;
        }
      } else { // Not positional.
        // No positional parameters can come after this parameter.
        allowPositionalNext = false;

        if (!isParamNamed(paramAnnot)) {
          errorf(param, "Parameter '%s' must be either positional or named", paramAnnot.name());
        }
      }
      if (isParamNamed(paramAnnot)) {
        // No positional-only parameters can come after this parameter.
        allowPositionalOnlyNext = false;
      }
    }

    if (annotation.extraPositionals().enableOnlyWithFlag() != FlagIdentifier.NONE
        || annotation.extraPositionals().disableWithFlag() != FlagIdentifier.NONE) {
      errorf(method, "The extraPositionals parameter may not be toggled by semantic flag");
    }
    if (annotation.extraKeywords().enableOnlyWithFlag() != FlagIdentifier.NONE
        || annotation.extraKeywords().disableWithFlag() != FlagIdentifier.NONE) {
      errorf(method, "The extraKeywords parameter may not be toggled by semantic flag");
    }
  }

  private static boolean isParamNamed(Param param) {
    return param.named() || param.legacyNamed();
  }

  // Checks consistency of a single parameter with its Param annotation.
  private void verifyParameter(Element param, Param paramAnnot, TypeMirror objectType) {
    TypeMirror paramType = param.asType(); // type of the Java method parameter

    // A "noneable" parameter variable must accept the value None.
    // A parameter whose default is None must be noneable.
    if (paramAnnot.noneable()) {
      if (!types.isSameType(paramType, objectType)) {
        errorf(
            param,
            "Expected type 'Object' but got type '%s' for noneable parameter '%s'. The argument"
                + " for a noneable parameter may be None, so the java parameter must be"
                + " compatible with the type of None as well as possible non-None values.",
            paramType,
            param.getSimpleName());
      }
    } else if (paramAnnot.defaultValue().equals("None")) {
      errorf(
          param,
          "Parameter '%s' has 'None' default value but is not noneable. (If this is intended"
              + " as a mandatory parameter, leave the defaultValue field empty)",
          paramAnnot.name());
    }

    // Check param.type.
    if (!types.isSameType(getParamType(paramAnnot), objectType)) {
      // Reject Param.type if not assignable to parameter variable.
      TypeMirror t = getParamType(paramAnnot);
      if (!types.isAssignable(t, types.erasure(paramType))) {
        errorf(
            param,
            "annotated type %s of parameter %s is not assignable to variable of type %s",
            t,
            paramAnnot.name(),
            paramType);
      }

      // Reject the combination of Param.type and Param.allowed_types.
      if (paramAnnot.allowedTypes().length > 0) {
        errorf(
            param,
            "Parameter '%s' has both 'type' and 'allowedTypes' specified. Only one may be"
                + " specified.",
            paramAnnot.name());
      }
    }

    // Reject an Param.allowed_type if not assignable to parameter variable.
    for (ParamType paramTypeAnnot : paramAnnot.allowedTypes()) {
      TypeMirror t = getParamTypeType(paramTypeAnnot);
      if (!types.isAssignable(t, types.erasure(paramType))) {
        errorf(
            param,
            "annotated allowed_type %s of parameter %s is not assignable to variable of type %s",
            t,
            paramAnnot.name(),
            paramType);
      }
    }

    // Reject generic types C<T> other than C<?>,
    // since reflective calls check only the toplevel class.
    if (paramType instanceof DeclaredType) {
      DeclaredType declaredType = (DeclaredType) paramType;
      for (TypeMirror typeArg : declaredType.getTypeArguments()) {
        if (!(typeArg instanceof WildcardType)) {
          errorf(
              param,
              "Parameter %s has generic type %s, but only wildcard type parameters are"
                  + " allowed. Type inference in a Starlark-exposed method is unsafe. See"
                  + " @SkylarkCallable class documentation for details.",
              param.getSimpleName(),
              paramType);
        }
      }
    }

    // Check sense of flag-controlled parameters.
    if (paramAnnot.enableOnlyWithFlag() != FlagIdentifier.NONE
        && paramAnnot.disableWithFlag() != FlagIdentifier.NONE) {
      errorf(
          param,
          "Parameter '%s' has enableOnlyWithFlag and disableWithFlag set. At most one may be set",
          paramAnnot.name());
    }
    boolean isParamControlledByFlag =
        paramAnnot.enableOnlyWithFlag() != FlagIdentifier.NONE
            || paramAnnot.disableWithFlag() != FlagIdentifier.NONE;
    if (!isParamControlledByFlag && !paramAnnot.valueWhenDisabled().isEmpty()) {
      errorf(
          param,
          "Parameter '%s' has valueWhenDisabled set, but is always enabled",
          paramAnnot.name());
    } else if (isParamControlledByFlag && paramAnnot.valueWhenDisabled().isEmpty()) {
      errorf(
          param,
          "Parameter '%s' may be disabled by semantic flag, thus valueWhenDisabled must be set",
          paramAnnot.name());
    }
  }

  // Returns the logical type of Param.type.
  private static TypeMirror getParamType(Param param) {
    // See explanation of this hack at Element.getAnnotation
    // and at https://stackoverflow.com/a/10167558.
    try {
      param.type();
      throw new IllegalStateException("unreachable");
    } catch (MirroredTypeException ex) {
      return ex.getTypeMirror();
    }
  }

  // Returns the logical type of ParamType.type.
  private static TypeMirror getParamTypeType(ParamType paramType) {
    // See explanation of this hack at Element.getAnnotation
    // and at https://stackoverflow.com/a/10167558.
    try {
      paramType.type();
      throw new IllegalStateException("unreachable");
    } catch (MirroredTypeException ex) {
      return ex.getTypeMirror();
    }
  }

  private void verifyExtraInterpreterParams(ExecutableElement methodElement,
      SkylarkCallable annotation) throws SkylarkCallableProcessorException {
    List<? extends VariableElement> methodSignatureParams = methodElement.getParameters();
    int currentIndex = methodSignatureParams.size() - numExpectedExtraInterpreterParams(annotation);

    // TODO(cparsons): Matching by class name alone is somewhat brittle, but due to tangled
    // dependencies, it is difficult for this processor to depend directly on the expected
    // classes here.
    if (!annotation.extraPositionals().name().isEmpty()) {
      if (!SKYLARK_LIST.equals(methodSignatureParams.get(currentIndex).asType().toString())) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Expected parameter index %d to be the %s type, matching extraPositionals, "
                    + "but was %s",
                currentIndex,
                SKYLARK_LIST,
                methodSignatureParams.get(currentIndex).asType().toString()));
      }
      currentIndex++;
    }
    if (!annotation.extraKeywords().name().isEmpty()) {
      if (!SKYLARK_DICT.equals(methodSignatureParams.get(currentIndex).asType().toString())) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Expected parameter index %d to be the %s type, matching extraKeywords, "
                    + "but was %s",
                currentIndex,
                SKYLARK_DICT,
                methodSignatureParams.get(currentIndex).asType().toString()));
      }
      currentIndex++;
    }
    if (annotation.useLocation()) {
      if (!LOCATION.equals(methodSignatureParams.get(currentIndex).asType().toString())) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Expected parameter index %d to be the %s type, matching useLocation, but was %s",
                currentIndex,
                LOCATION,
                methodSignatureParams.get(currentIndex).asType().toString()));
      }
      currentIndex++;
    }
    if (annotation.useStarlarkThread()) {
      if (!STARLARK_THREAD.equals(methodSignatureParams.get(currentIndex).asType().toString())) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Expected parameter index %d to be the %s type, matching useStarlarkThread, "
                    + "but was %s",
                currentIndex,
                STARLARK_THREAD,
                methodSignatureParams.get(currentIndex).asType().toString()));
      }
      currentIndex++;
    }
    if (annotation.useStarlarkSemantics()) {
      if (!STARLARK_SEMANTICS.equals(methodSignatureParams.get(currentIndex).asType().toString())) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Expected parameter index %d to be the %s type, matching useStarlarkSemantics, "
                    + "but was %s",
                currentIndex,
                STARLARK_SEMANTICS,
                methodSignatureParams.get(currentIndex).asType()));
      }
      currentIndex++;
    }
  }

  private int numExpectedExtraInterpreterParams(SkylarkCallable annotation) {
    int numExtraInterpreterParams = 0;
    numExtraInterpreterParams += annotation.extraPositionals().name().isEmpty() ? 0 : 1;
    numExtraInterpreterParams += annotation.extraKeywords().name().isEmpty() ? 0 : 1;
    numExtraInterpreterParams += annotation.useLocation() ? 1 : 0;
    numExtraInterpreterParams += annotation.useStarlarkThread() ? 1 : 0;
    numExtraInterpreterParams += annotation.useStarlarkSemantics() ? 1 : 0;
    return numExtraInterpreterParams;
  }

  /**
   * Prints an error message & fails the compilation.
   *
   * @param e The element which has caused the error. Can be null
   * @param msg The error message
   */
  private void error(Element e, String msg) {
    messager.printMessage(Diagnostic.Kind.ERROR, msg, e);
  }

  // A variant of 'error' that formats the error message.
  @FormatMethod
  private void errorf(Element e, String format, Object... args) {
    messager.printMessage(Diagnostic.Kind.ERROR, String.format(format, args), e);
  }

  private static class SkylarkCallableProcessorException extends Exception {
    private final Element element;
    private final String errorMessage;

    private SkylarkCallableProcessorException(Element element, String errorMessage) {
      this.element = element;
      this.errorMessage = errorMessage;
    }
  }
}
