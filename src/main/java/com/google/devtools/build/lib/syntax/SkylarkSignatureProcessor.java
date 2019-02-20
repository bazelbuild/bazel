// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Booleans;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction.ExtraArgKind;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * This class defines utilities to process @SkylarkSignature annotations
 * to configure a given field.
 */
public class SkylarkSignatureProcessor {

  // A cache mapping string representation of a skylark parameter default value to the object
  // represented by that string. For example, "None" -> Runtime.NONE. This cache is manually
  // maintained (instead of using, for example, a LoadingCache), as default values may sometimes
  // be recursively requested.
  private static final ConcurrentHashMap<String, Object> defaultValueCache =
      new ConcurrentHashMap<>();

  /**
   * Extracts a {@code FunctionSignature.WithValues<Object, SkylarkType>} from a
   * {@link SkylarkCallable}-annotated method.
   *
   * @param name the name of the function
   * @param descriptor the method descriptor
   * @param paramDoc an optional list into which to store documentation strings
   * @param enforcedTypesList an optional list into which to store effective types to enforce
   */
  public static FunctionSignature.WithValues<Object, SkylarkType> getSignatureForCallable(
      String name, MethodDescriptor descriptor,
      @Nullable List<String> paramDoc, @Nullable List<SkylarkType> enforcedTypesList) {

    SkylarkCallable annotation = descriptor.getAnnotation();

    // TODO(cparsons): Validate these properties with the annotation processor instead.
    Preconditions.checkArgument(name.equals(annotation.name()),
        "%s != %s", name, annotation.name());
    boolean documented = annotation.documented();
    if (annotation.doc().isEmpty() && documented) {
      throw new RuntimeException(String.format("function %s is undocumented", name));
    }

    return getSignatureForCallable(
        name,
        documented,
        annotation.parameters(),
        annotation.extraPositionals(),
        annotation.extraKeywords(),
        /*defaultValues=*/ null,
        paramDoc,
        enforcedTypesList);
  }


  /**
   * Extracts a {@code FunctionSignature.WithValues<Object, SkylarkType>} from a
   * {@link SkylarkSignature} annotation.
   *
   * @param name the name of the function
   * @param annotation the annotation
   * @param defaultValues an optional list of default values
   * @param paramDoc an optional list into which to store documentation strings
   * @param enforcedTypesList an optional list into which to store effective types to enforce
   */
  // NB: the two arguments paramDoc and enforcedTypesList are used to "return" extra values via
  // side-effects, and that's ugly
  // TODO(bazel-team): use AutoValue to declare a value type to use as return value?
  public static FunctionSignature.WithValues<Object, SkylarkType> getSignatureForCallable(
      String name, SkylarkSignature annotation,
      @Nullable Iterable<Object> defaultValues,
      @Nullable List<String> paramDoc, @Nullable List<SkylarkType> enforcedTypesList) {

    Preconditions.checkArgument(name.equals(annotation.name()),
        "%s != %s", name, annotation.name());
    boolean documented = annotation.documented();
    if (annotation.doc().isEmpty() && documented) {
      throw new RuntimeException(String.format("function %s is undocumented", name));
    }
    return getSignatureForCallable(name, documented,
        annotation.parameters(),
        annotation.extraPositionals(),
        annotation.extraKeywords(), defaultValues, paramDoc, enforcedTypesList);
  }

  private static boolean isParamNamed(Param param) {
    return param.named() || param.legacyNamed();
  }

  private static FunctionSignature.WithValues<Object, SkylarkType> getSignatureForCallable(
      String name, boolean documented,
      Param[] parameters,
      @Nullable Param extraPositionals, @Nullable Param extraKeywords,
      @Nullable Iterable<Object> defaultValues,
      @Nullable List<String> paramDoc, @Nullable List<SkylarkType> enforcedTypesList) {
    ArrayList<Parameter<Object, SkylarkType>> paramList = new ArrayList<>();
    HashMap<String, SkylarkType> enforcedTypes =
        enforcedTypesList == null ? null : new HashMap<>();

    HashMap<String, String> doc = new HashMap<>();

    Iterator<Object> defaultValuesIterator = defaultValues == null
        ? null : defaultValues.iterator();
    try {
      boolean named = false;
      for (Param param : parameters) {
        boolean mandatory = param.defaultValue() != null && param.defaultValue().isEmpty();
        Object defaultValue = mandatory ? null : getDefaultValue(param, defaultValuesIterator);
        if (isParamNamed(param) && !param.positional() && !named) {
          named = true;
          @Nullable Param starParam = null;
          if (extraPositionals != null && !extraPositionals.name().isEmpty()) {
            starParam = extraPositionals;
          }
          paramList.add(getParameter(name, starParam, enforcedTypes, doc, documented,
                /*mandatory=*/false, /*star=*/true, /*starStar=*/false, /*defaultValue=*/null));
        }
        paramList.add(getParameter(name, param, enforcedTypes, doc, documented,
                mandatory, /*star=*/false, /*starStar=*/false, defaultValue));
      }
      if (extraPositionals != null && !extraPositionals.name().isEmpty() && !named) {
        paramList.add(getParameter(name, extraPositionals, enforcedTypes, doc,
            documented, /*mandatory=*/false, /*star=*/true, /*starStar=*/false,
            /*defaultValue=*/null));
      }
      if (extraKeywords != null && !extraKeywords.name().isEmpty()) {
        paramList.add(
            getParameter(name, extraKeywords, enforcedTypes, doc, documented,
                /*mandatory=*/false, /*star=*/false, /*starStar=*/true, /*defaultValue=*/null));
      }
      FunctionSignature.WithValues<Object, SkylarkType> signature =
          FunctionSignature.WithValues.of(paramList);
      for (String paramName : signature.getSignature().getNames()) {
        if (enforcedTypesList != null) {
          enforcedTypesList.add(enforcedTypes.get(paramName));
        }
        if (paramDoc != null) {
          paramDoc.add(doc.get(paramName));
        }
      }
      return signature;
    } catch (FunctionSignature.SignatureException e) {
      throw new RuntimeException(String.format(
          "Invalid signature while configuring BuiltinFunction %s", name), e);
    }
  }

  /**
   * Configures the parameter of this Skylark function using the annotation.
   */
  // TODO(bazel-team): Maybe have the annotation be a string representing the
  // python-style calling convention including default values, and have the regular Parser
  // process it? (builtin function call not allowed when evaluating values, but more complex
  // values are possible by referencing variables in some definition environment).
  // Then the only per-parameter information needed is a documentation string.
  private static Parameter<Object, SkylarkType> getParameter(
      String name, Param param, Map<String, SkylarkType> enforcedTypes,
      Map<String, String> paramDoc, boolean documented,
      boolean mandatory, boolean star, boolean starStar, @Nullable Object defaultValue)
      throws FunctionSignature.SignatureException {

    @Nullable SkylarkType officialType = null;
    @Nullable SkylarkType enforcedType = null;
    if (star && param == null) { // pseudo-parameter to separate positional from named-only
      return new Parameter.Star<>(null);
    }
    if (param.type() != Object.class) {
      if (param.generic1() != Object.class) {
        // Enforce the proper parametric type for Skylark list and set objects
        officialType = SkylarkType.of(param.type(), param.generic1());
        enforcedType = officialType;
      } else {
        officialType = SkylarkType.of(param.type());
        enforcedType = officialType;
      }
      if (param.callbackEnabled()) {
        officialType = SkylarkType.Union.of(
            officialType, SkylarkType.SkylarkFunctionType.of(name, officialType));
        enforcedType = SkylarkType.Union.of(
            enforcedType, SkylarkType.SkylarkFunctionType.of(name, enforcedType));
      }
      if (param.noneable()) {
        officialType = SkylarkType.Union.of(officialType, SkylarkType.NONE);
        enforcedType = SkylarkType.Union.of(enforcedType, SkylarkType.NONE);
      }
    }
    if (enforcedTypes != null) {
      enforcedTypes.put(param.name(), enforcedType);
    }
    if (param.doc().isEmpty() && documented) {
      throw new RuntimeException(
          String.format("parameter %s on method %s is undocumented", param.name(), name));
    }
    if (paramDoc != null) {
      paramDoc.put(param.name(), param.doc());
    }
    if (starStar) {
      return new Parameter.StarStar<>(Identifier.of(param.name()), officialType);
    } else if (star) {
      return new Parameter.Star<>(Identifier.of(param.name()), officialType);
    } else if (mandatory) {
      return new Parameter.Mandatory<>(Identifier.of(param.name()), officialType);
    } else if (defaultValue != null
        && !defaultValue.equals(Runtime.UNBOUND)
        && enforcedType != null) {
      Preconditions.checkArgument(enforcedType.contains(defaultValue),
          "In function '%s', parameter '%s' has default value %s that isn't of enforced type %s",
          name, param.name(), Printer.repr(defaultValue), enforcedType);
    }
    return new Parameter.Optional<>(Identifier.of(param.name()), officialType, defaultValue);
  }

  static Object getDefaultValue(Param param, Iterator<Object> iterator) {
    return getDefaultValue(param.name(), param.defaultValue(), iterator);
  }

  static Object getDefaultValue(
      String paramName, String paramDefaultValue, Iterator<Object> iterator) {
    if (iterator != null) {
      return iterator.next();
    } else if (paramDefaultValue.isEmpty()) {
      return Runtime.NONE;
    } else {
      try {
        Object defaultValue = defaultValueCache.get(paramDefaultValue);
        if (defaultValue != null) {
          return defaultValue;
        }
        try (Mutability mutability = Mutability.create("initialization")) {
          // Note that this Skylark environment ignores command line flags.
          Environment env =
              Environment.builder(mutability)
                  .useDefaultSemantics()
                  .setGlobals(Environment.CONSTANTS_ONLY)
                  .setEventHandler(Environment.FAIL_FAST_HANDLER)
                  .build()
                  .update("unbound", Runtime.UNBOUND);
          defaultValue = BuildFileAST.eval(env, paramDefaultValue);
          defaultValueCache.put(paramDefaultValue, defaultValue);
          return defaultValue;
        }
      } catch (Exception e) {
        throw new RuntimeException(
            String.format(
                "Exception while processing @SkylarkSignature.Param %s, default value %s",
                paramName, paramDefaultValue),
            e);
      }
    }
  }

  /** Extract additional signature information for BuiltinFunction-s */
  public static ExtraArgKind[] getExtraArgs(SkylarkSignature annotation) {
    final int numExtraArgs =
        Booleans.countTrue(
            annotation.useLocation(), annotation.useAst(), annotation.useEnvironment());
    if (numExtraArgs == 0) {
      return null;
    }
    final ExtraArgKind[] extraArgs = new ExtraArgKind[numExtraArgs];
    int i = 0;
    if (annotation.useLocation()) {
      extraArgs[i++] = ExtraArgKind.LOCATION;
    }
    if (annotation.useAst()) {
      extraArgs[i++] = ExtraArgKind.SYNTAX_TREE;
    }
    if (annotation.useEnvironment()) {
      extraArgs[i++] = ExtraArgKind.ENVIRONMENT;
    }
    return extraArgs;
  }

  /**
   * Processes all {@link SkylarkSignature}-annotated fields in a class.
   *
   * <p>This includes registering these fields as builtins using {@link Runtime}, and for {@link
   * BaseFunction} instances, calling {@link BaseFunction#configure(SkylarkSignature)}. The fields
   * will be picked up by reflection even if they are not public.
   *
   * <p>This function should be called once per class, before the builtins registry is frozen. In
   * practice, this is usually called from the class's own static initializer block. E.g., a class
   * {@code Foo} containing {@code @SkylarkSignature} annotations would end with
   * {@code static { SkylarkSignatureProcessor.configureSkylarkFunctions(Foo.class); }}.
   *
   * <p><b>If you see exceptions from {@link Runtime.BuiltinRegistry} here:</b> Be sure the class's
   * static initializer has in fact been called before the registry was frozen. In Bazel, see
   * {@link com.google.devtools.build.lib.runtime.BlazeRuntime#initSkylarkBuiltinsRegistry}.
   */
  public static void configureSkylarkFunctions(Class<?> type) {
    Runtime.BuiltinRegistry builtins = Runtime.getBuiltinRegistry();
    for (Field field : type.getDeclaredFields()) {
      if (field.isAnnotationPresent(SkylarkSignature.class)) {
        // The annotated fields are often private, but we need access them.
        field.setAccessible(true);
        SkylarkSignature annotation = field.getAnnotation(SkylarkSignature.class);
        Object value = null;
        try {
          value =
              Preconditions.checkNotNull(
                  field.get(null),
                  "Error while trying to configure %s.%s: its value is null",
                  type,
                  field);
          builtins.registerBuiltin(type, field.getName(), value);
          if (BaseFunction.class.isAssignableFrom(field.getType())) {
            BaseFunction function = (BaseFunction) value;
            if (!function.isConfigured()) {
              function.configure(annotation);
            }
            Class<?> nameSpace = function.getObjectType();
            if (nameSpace != null) {
              Preconditions.checkState(!(function instanceof BuiltinFunction.Factory));
              builtins.registerFunction(nameSpace, function);
            }
          }
        } catch (IllegalAccessException e) {
          throw new RuntimeException(String.format(
              "Error while trying to configure %s.%s (value %s)", type, field, value), e);
        }
      }
    }
  }
}
