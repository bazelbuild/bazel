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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
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
 *       <pre>method([positionals]*[other user-args](Location)(FuncallExpression)(StarlarkThread))
 *       </pre>
 *       where Location, FuncallExpression, and StarlarkThread are supplied by the interpreter if
 *       and only if useLocation, useAst, and useStarlarkThread are specified, respectively.
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
  private static final String AST = "com.google.devtools.build.lib.syntax.FuncallExpression";
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

  private TypeMirror getType(String name) {
    return elements.getTypeElement(name).asType();
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
        error(
            cls,
            String.format(
                "class %s has @SkylarkModule annotation but does not implement StarlarkValue",
                cls.getSimpleName()));
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
        verifyParamSemantics(methodElement, annotation);
        verifyParamFlagSemantics(methodElement, annotation);
        verifyParamGenericTypes(methodElement);
        verifyNumberOfParameters(methodElement, annotation);
        verifyExtraInterpreterParams(methodElement, annotation);
        verifyIfSelfCall(methodElement, annotation);
        verifyFlagToggles(methodElement, annotation);
        verifyNoNameConflict(methodElement, annotation);
      } catch (SkylarkCallableProcessorException exception) {
        // TODO(adonovan): don't use exceptions; report multiple errors per pass
        // as this saves time in compiler-driven refactoring.
        error(exception.methodElement, exception.errorMessage);
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
          error(
              methodElement,
              String.format(
                  "@SkylarkCallable-annotated method %s returns %s, which has no legal Starlark"
                      + " values (see Starlark.fromJava)",
                  methodElement.getSimpleName(), ret));
        }
      }

      // Check that the method's class is SkylarkGlobalLibrary-annotated,
      // or implements StarlarkValue, or an error has already been reported.
      Element cls = methodElement.getEnclosingElement();
      if (okClasses.add(cls) && !types.isAssignable(cls.asType(), skylarkValueType)) {
        error(
            cls,
            String.format(
                "method %s has @SkylarkCallable annotation but enclosing class %s does not"
                    + " implement StarlarkValue nor has @SkylarkGlobalLibrary annotation",
                methodElement.getSimpleName(), cls.getSimpleName()));
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
      if (annotation.useAst()
          || annotation.useStarlarkThread()
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
                + "useAst, useStarlarkThread, extraPositionals, or extraKeywords");
      }
    }
  }

  private static boolean isParamNamed(Param param) {
    return param.named() || param.legacyNamed();
  }

  private void verifyParamSemantics(ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {
    boolean allowPositionalNext = true;
    boolean allowPositionalOnlyNext = true;
    boolean allowNonDefaultPositionalNext = true;

    int paramIndex = 0;
    for (Param parameter : annotation.parameters()) {
      if (parameter.noneable()) {
        VariableElement methodParam = methodElement.getParameters().get(paramIndex);
        if (!"java.lang.Object".equals(methodParam.asType().toString())) {
          throw new SkylarkCallableProcessorException(
              methodElement,
              String.format(
                  "Expected type 'Object' but got type '%s' for noneable parameter '%s'. The "
                      + "argument for a noneable parameter may be None, so the java parameter "
                      + "must be compatible with the type of None as well as possible non-None "
                      + "values.",
                  methodParam.asType(), methodParam.getSimpleName()));
        }
      } else { // !parameter.noneable()
        if ("None".equals(parameter.defaultValue())) {
          throw new SkylarkCallableProcessorException(
              methodElement,
              String.format(
                  "Parameter '%s' has 'None' default value but is not noneable. "
                      + "(If this is intended as a mandatory parameter, leave the defaultValue "
                      + "field empty)",
                  parameter.name()));
        }
      }

      if (!parameter.positional() && !parameter.named()) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format("Parameter '%s' must be either positional or named",
                parameter.name()));
      }
      if ((parameter.allowedTypes().length > 0)
          && (!"java.lang.Object".equals(paramTypeFieldCanonicalName(parameter)))) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format("Parameter '%s' has both 'type' and 'allowedTypes' specified. Only"
                    + " one may be specified.",
                parameter.name()));
      }

      if (parameter.positional()) {
        if (!allowPositionalNext) {
          throw new SkylarkCallableProcessorException(
              methodElement,
              String.format(
                  "Positional parameter '%s' is specified after one or more "
                      + "non-positonal parameters",
                  parameter.name()));
        }
        if (!isParamNamed(parameter) && !allowPositionalOnlyNext) {
          throw new SkylarkCallableProcessorException(
              methodElement,
              String.format(
                  "Positional-only parameter '%s' is specified after one or more "
                      + "named parameters",
                  parameter.name()));
        }
        if (parameter.defaultValue().isEmpty()) { // There is no default value.
          if (!allowNonDefaultPositionalNext) {
            throw new SkylarkCallableProcessorException(
                methodElement,
                String.format(
                    "Positional parameter '%s' has no default value but is specified after one "
                        + "or more positional parameters with default values",
                    parameter.name()));
          }
        } else { // There is a default value.
          // No positional parameters without a default value can come after this parameter.
          allowNonDefaultPositionalNext = false;
        }
      } else { // Not positional.
        // No positional parameters can come after this parameter.
        allowPositionalNext = false;
      }
      if (isParamNamed(parameter)) {
        // No positional-only parameters can come after this parameter.
        allowPositionalOnlyNext = false;
      }
      paramIndex++;
    }
  }

  private void verifyParamFlagSemantics(ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {

    for (Param parameter : annotation.parameters()) {
      if (parameter.enableOnlyWithFlag() != FlagIdentifier.NONE
          && parameter.disableWithFlag() != FlagIdentifier.NONE) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Parameter '%s' has enableOnlyWithFlag and disableWithFlag set. "
                    + "At most one may be set",
                parameter.name()));
      }

      boolean isParamControlledByFlag =
          parameter.enableOnlyWithFlag() != FlagIdentifier.NONE
              || parameter.disableWithFlag() != FlagIdentifier.NONE;

      if (!isParamControlledByFlag && !parameter.valueWhenDisabled().isEmpty()) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Parameter '%s' has valueWhenDisabled set, but is always enabled",
                parameter.name()));
      } else if (isParamControlledByFlag && parameter.valueWhenDisabled().isEmpty()) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Parameter '%s' may be disabled by semantic flag, "
                    + "thus valueWhenDisabled must be set",
                parameter.name()));
      }
    }

    if (annotation.extraPositionals().enableOnlyWithFlag() != FlagIdentifier.NONE
        || annotation.extraPositionals().disableWithFlag() != FlagIdentifier.NONE) {
      throw new SkylarkCallableProcessorException(
          methodElement, "The extraPositionals parameter may not be toggled by semantic flag");
    }
    if (annotation.extraKeywords().enableOnlyWithFlag() != FlagIdentifier.NONE
        || annotation.extraKeywords().disableWithFlag() != FlagIdentifier.NONE) {
      throw new SkylarkCallableProcessorException(
          methodElement, "The extraKeywords parameter may not be toggled by semantic flag");
    }
  }

  private String paramTypeFieldCanonicalName(Param param) {
    try {
      return param.type().toString();
    } catch (MirroredTypeException exception) {
      // This is a hack to obtain the actual canonical name of param.type(). Doing this ths
      // "correct" way results in far less readable code.
      // Since this processor is only for compile-time checks, this isn't efficiency we need
      // to worry about.
      return exception.getTypeMirror().toString();
    }
  }

  private void verifyNumberOfParameters(ExecutableElement methodElement, SkylarkCallable annotation)
      throws SkylarkCallableProcessorException {
    List<? extends VariableElement> methodSignatureParams = methodElement.getParameters();
    int numExtraInterpreterParams = numExpectedExtraInterpreterParams(annotation);

    int numDeclaredArgs = annotation.parameters().length;
    if (methodSignatureParams.size() != numDeclaredArgs + numExtraInterpreterParams) {
      throw new SkylarkCallableProcessorException(
          methodElement,
          String.format(
              "@SkylarkCallable annotated method has %d parameters, but annotation declared "
                  + "%d user-supplied parameters and %d extra interpreter parameters.",
              methodSignatureParams.size(), numDeclaredArgs, numExtraInterpreterParams));
    }
    if (annotation.structField()) {
      if (methodSignatureParams.size() != numExtraInterpreterParams) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "@SkylarkCallable annotated methods with structField=true must have "
                    + "0 user-supplied parameters. Expected %d extra interpreter parameters, "
                    + "but found %d total parameters.",
                numExtraInterpreterParams, methodSignatureParams.size()));
      }
    }
  }

  private static void verifyParamGenericTypes(ExecutableElement methodElement)
      throws SkylarkCallableProcessorException {
    for (VariableElement methodParam : methodElement.getParameters()) {
      if (methodParam.asType() instanceof DeclaredType) {
        DeclaredType declaredType = (DeclaredType) methodParam.asType();
        for (TypeMirror typeArg : declaredType.getTypeArguments()) {
          if (!(typeArg instanceof WildcardType)) {
            throw new SkylarkCallableProcessorException(
                methodElement,
                String.format(
                    "Parameter %s has generic type %s, but may only wildcard type parameters are "
                        + "allowed. Type inference in a Starlark-exposed method is unsafe. See "
                        + "@SkylarkCallable class documentation for details.",
                    methodParam.getSimpleName(), methodParam.asType()));
          }
        }
      }
    }
  }

  private void verifyExtraInterpreterParams(ExecutableElement methodElement,
      SkylarkCallable annotation) throws SkylarkCallableProcessorException {
    List<? extends VariableElement> methodSignatureParams = methodElement.getParameters();
    int currentIndex = methodSignatureParams.size() - numExpectedExtraInterpreterParams(annotation);

    // TODO(cparsons): Matching by class name alone is somewhat brittle, but due to tangled
    // dependencies, it is difficult for this processor to depend directy on the expected
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
    if (annotation.useAst()) {
      if (!AST.equals(methodSignatureParams.get(currentIndex).asType().toString())) {
        throw new SkylarkCallableProcessorException(
            methodElement,
            String.format(
                "Expected parameter index %d to be the %s type, matching useAst, but was %s",
                currentIndex, AST, methodSignatureParams.get(currentIndex).asType().toString()));
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
    numExtraInterpreterParams += annotation.useAst() ? 1 : 0;
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

  private static class SkylarkCallableProcessorException extends Exception {
    private final ExecutableElement methodElement;
    private final String errorMessage;

    private SkylarkCallableProcessorException(
        ExecutableElement methodElement, String errorMessage) {
      this.methodElement = methodElement;
      this.errorMessage = errorMessage;
    }
  }
}
