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
 * Annotation processor for {@link SkylarkCallable}. See that class for requirements.
 *
 * <p>These properties can be relied upon at runtime without additional checks.
 */
@SupportedAnnotationTypes({
  "com.google.devtools.build.lib.skylarkinterface.SkylarkCallable",
  "com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary",
  "com.google.devtools.build.lib.skylarkinterface.SkylarkModule"
})
public final class SkylarkCallableProcessor extends AbstractProcessor {

  private Types types;
  private Elements elements;
  private Messager messager;

  // A set containing a TypeElement for each class with a SkylarkCallable.selfCall annotation.
  private Set<Element> classesWithSelfcall;
  // A multimap where keys are class element, and values are the callable method names identified in
  // that class (where "method name" is SkylarkCallable.name).
  private SetMultimap<Element, String> processedClassMethods;

  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported();
  }

  @Override
  public synchronized void init(ProcessingEnvironment env) {
    super.init(env);
    this.types = env.getTypeUtils();
    this.elements = env.getElementUtils();
    this.messager = env.getMessager();
    this.classesWithSelfcall = new HashSet<>();
    this.processedClassMethods = LinkedHashMultimap.create();
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
            "class %s has SkylarkModule annotation but does not implement StarlarkValue",
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
      // Only methods are annotated with SkylarkCallable.
      // This is ensured by the @Target(ElementType.METHOD) annotation.
      ExecutableElement method = (ExecutableElement) element;
      if (!method.getModifiers().contains(Modifier.PUBLIC)) {
        errorf(method, "SkylarkCallable-annotated methods must be public.");
      }
      if (method.getModifiers().contains(Modifier.STATIC)) {
        errorf(method, "SkylarkCallable-annotated methods cannot be static.");
      }

      // Check the annotation itself.
      SkylarkCallable annot = method.getAnnotation(SkylarkCallable.class);
      if (annot.name().isEmpty()) {
        errorf(method, "SkylarkCallable.name must be non-empty.");
      }
      Element cls = method.getEnclosingElement();
      if (!processedClassMethods.put(cls, annot.name())) {
        errorf(method, "Containing class defines more than one method named '%s'.", annot.name());
      }
      if (annot.documented() && annot.doc().isEmpty()) {
        errorf(method, "The 'doc' string must be non-empty if 'documented' is true.");
      }
      if (annot.structField()) {
        checkStructFieldAnnotation(method, annot);
      } else if (annot.useStarlarkSemantics()) {
        errorf(
            method,
            "a SkylarkCallable-annotated method with structField=false may not also specify"
                + " useStarlarkSemantics. (Instead, set useStarlarkThread and call"
                + " getSemantics().)");
      }
      if (annot.selfCall() && !classesWithSelfcall.add(cls)) {
        errorf(method, "Containing class has more than one selfCall method defined.");
      }
      if (annot.enableOnlyWithFlag() != FlagIdentifier.NONE
          && annot.disableWithFlag() != FlagIdentifier.NONE) {
        errorf(
            method,
            "Only one of SkylarkCallable.enablingFlag and SkylarkCallable.disablingFlag may be"
                + " specified.");
      }

      checkParameters(method, annot);

      // Verify that result type, if final, might satisfy Starlark.fromJava.
      // (If the type is non-final we can't prove that all subclasses are invalid.)
      TypeMirror ret = method.getReturnType();
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
              method,
              "SkylarkCallable-annotated method %s returns %s, which has no legal Starlark values"
                  + " (see Starlark.fromJava)",
              method.getSimpleName(),
              ret);
        }
      }

      // Check that the method's class is SkylarkGlobalLibrary-annotated,
      // or implements StarlarkValue, or an error has already been reported.
      if (okClasses.add(cls) && !types.isAssignable(cls.asType(), skylarkValueType)) {
        errorf(
            cls,
            "method %s has SkylarkCallable annotation but enclosing class %s does not implement"
                + " StarlarkValue nor has SkylarkGlobalLibrary annotation",
            method.getSimpleName(),
            cls.getSimpleName());
      }
    }

    // Returning false allows downstream processors to work on the same annotations
    return false;
  }

  // TODO(adonovan): obviate these checks by separating field/method interfaces.
  private void checkStructFieldAnnotation(ExecutableElement method, SkylarkCallable annot) {
    // useStructField is incompatible with special thread-related parameters,
    // because unlike a method, which is actively called within a thread,
    // a field is a passive part of a data structure that may be accessed
    // from Java threads that don't have anything to do with Starlark threads.
    // However, the StarlarkSemantics is available even to fields,
    // because it is a required parameter for all attribute-selection
    // operations x.f.
    //
    // Not having a thread forces implementations to assume Mutability=null,
    // which is not quite right. Perhaps one day we can abolish Mutability
    // in favor of a tracing approach as in go.starlark.net.
    if (annot.useStarlarkThread()) {
      errorf(
          method,
          "a SkylarkCallable-annotated method with structField=true may not also specify"
              + " useStarlarkThread");
    }
    if (annot.useLocation()) {
      errorf(
          method,
          "a SkylarkCallable-annotated method with structField=true may not also specify"
              + " useLocation");
    }

    if (!annot.extraPositionals().name().isEmpty()) {
      errorf(
          method,
          "a SkylarkCallable-annotated method with structField=true may not also specify"
              + " extraPositionals");
    }
    if (!annot.extraKeywords().name().isEmpty()) {
      errorf(
          method,
          "a SkylarkCallable-annotated method with structField=true may not also specify"
              + " extraKeywords");
    }
    if (annot.selfCall()) {
      errorf(
          method,
          "a SkylarkCallable-annotated method with structField=true may not also specify"
              + " selfCall=true");
    }
    int nparams = annot.parameters().length;
    if (nparams > 0) {
      errorf(
          method,
          "method %s is annotated structField=true but also has %d Param annotations",
          method.getSimpleName(),
          nparams);
    }
  }

  private void checkParameters(ExecutableElement method, SkylarkCallable annot) {
    List<? extends VariableElement> params = method.getParameters();

    TypeMirror objectType = getType("java.lang.Object");

    boolean allowPositionalNext = true;
    boolean allowPositionalOnlyNext = true;
    boolean allowNonDefaultPositionalNext = true;

    // Check @Param annotations match parameters.
    Param[] paramAnnots = annot.parameters();
    for (int i = 0; i < paramAnnots.length; i++) {
      Param paramAnnot = paramAnnots[i];
      if (i >= params.size()) {
        errorf(
            method,
            "method %s has %d Param annotations but only %d parameters",
            method.getSimpleName(),
            paramAnnots.length,
            params.size());
        return;
      }
      VariableElement param = params.get(i);

      checkParameter(param, paramAnnot, objectType);

      // Check parameter ordering.
      if (paramAnnot.positional()) {
        if (!allowPositionalNext) {
          errorf(
              param,
              "Positional parameter '%s' is specified after one or more non-positional parameters",
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

    checkSpecialParams(method, annot);
  }

  private static boolean isParamNamed(Param param) {
    return param.named() || param.legacyNamed();
  }

  // Checks consistency of a single parameter with its Param annotation.
  private void checkParameter(Element param, Param paramAnnot, TypeMirror objectType) {
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
            "annotated type %s of parameter '%s' is not assignable to variable of type %s",
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
            "annotated allowed_type %s of parameter '%s' is not assignable to variable of type %s",
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
              "parameter '%s' has generic type %s, but only wildcard type parameters are"
                  + " allowed. Type inference in a Starlark-exposed method is unsafe. See"
                  + " SkylarkCallable class documentation for details.",
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

  private void checkSpecialParams(ExecutableElement method, SkylarkCallable annot) {
    if (annot.extraPositionals().enableOnlyWithFlag() != FlagIdentifier.NONE
        || annot.extraPositionals().disableWithFlag() != FlagIdentifier.NONE) {
      errorf(method, "The extraPositionals parameter may not be toggled by semantic flag");
    }
    if (annot.extraKeywords().enableOnlyWithFlag() != FlagIdentifier.NONE
        || annot.extraKeywords().disableWithFlag() != FlagIdentifier.NONE) {
      errorf(method, "The extraKeywords parameter may not be toggled by semantic flag");
    }

    List<? extends VariableElement> params = method.getParameters();
    int index = annot.parameters().length;

    // insufficient parameters?
    int special = numExpectedSpecialParams(annot);
    if (index + special > params.size()) {
      errorf(
          method,
          "method %s is annotated with %d Params plus %d special parameters, but has only %d"
              + " parameter variables",
          method.getSimpleName(),
          index,
          special,
          params.size());
      return; // not safe to proceed
    }

    if (!annot.extraPositionals().name().isEmpty()) {
      VariableElement param = params.get(index++);
      // Allow any supertype of Tuple<Object>.
      TypeMirror tupleOfObjectType =
          types.getDeclaredType(
              elements.getTypeElement("com.google.devtools.build.lib.syntax.Tuple"),
              getType("java.lang.Object"));
      if (!types.isAssignable(tupleOfObjectType, param.asType())) {
        errorf(
            param,
            "extraPositionals special parameter '%s' has type %s, to which Tuple<Object> cannot be"
                + " assigned",
            param.getSimpleName(),
            param.asType());
      }
    }

    if (!annot.extraKeywords().name().isEmpty()) {
      VariableElement param = params.get(index++);
      // Allow any supertype of Dict<String, Object>.
      TypeMirror dictOfStringObjectType =
          types.getDeclaredType(
              elements.getTypeElement("com.google.devtools.build.lib.syntax.Dict"),
              getType("java.lang.String"),
              getType("java.lang.Object"));
      if (!types.isAssignable(dictOfStringObjectType, param.asType())) {
        errorf(
            param,
            "extraKeywords special parameter '%s' has type %s, to which Dict<String, Object>"
                + " cannot be assigned",
            param.getSimpleName(),
            param.asType());
      }
    }

    if (annot.useLocation()) {
      VariableElement param = params.get(index++);
      TypeMirror locationType = getType("com.google.devtools.build.lib.events.Location");
      if (!types.isSameType(locationType, param.asType())) {
        errorf(
            param,
            "for useLocation special parameter '%s', got type %s, want Location",
            param.getSimpleName(),
            param.asType());
      }
    }

    if (annot.useStarlarkThread()) {
      VariableElement param = params.get(index++);
      TypeMirror threadType = getType("com.google.devtools.build.lib.syntax.StarlarkThread");
      if (!types.isSameType(threadType, param.asType())) {
        errorf(
            param,
            "for useStarlarkThread special parameter '%s', got type %s, want StarlarkThread",
            param.getSimpleName(),
            param.asType());
      }
    }

    if (annot.useStarlarkSemantics()) {
      VariableElement param = params.get(index++);
      TypeMirror semanticsType = getType("com.google.devtools.build.lib.syntax.StarlarkSemantics");
      if (!types.isSameType(semanticsType, param.asType())) {
        errorf(
            param,
            "for useStarlarkSemantics special parameter '%s', got type %s, want StarlarkSemantics",
            param.getSimpleName(),
            param.asType());
      }
    }

    // surplus parameters?
    if (index < params.size()) {
      errorf(
          params.get(index), // first surplus parameter
          "method %s is annotated with %d Params plus %d special parameters, yet has %d parameter"
              + " variables",
          method.getSimpleName(),
          annot.parameters().length,
          special,
          params.size());
    }
  }

  private static int numExpectedSpecialParams(SkylarkCallable annot) {
    int n = 0;
    n += annot.extraPositionals().name().isEmpty() ? 0 : 1;
    n += annot.extraKeywords().name().isEmpty() ? 0 : 1;
    n += annot.useLocation() ? 1 : 0;
    n += annot.useStarlarkThread() ? 1 : 0;
    n += annot.useStarlarkSemantics() ? 1 : 0;
    return n;
  }

  // Reports a (formatted) error and fails the compilation.
  @FormatMethod
  private void errorf(Element e, String format, Object... args) {
    messager.printMessage(Diagnostic.Kind.ERROR, String.format(format, args), e);
  }
}
