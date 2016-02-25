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

import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature.Param;
import com.google.devtools.build.lib.syntax.BuiltinFunction.ExtraArgKind;
import com.google.devtools.build.lib.util.Preconditions;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * This class defines utilities to process @SkylarkSignature annotations
 * to configure a given field.
 */
public class SkylarkSignatureProcessor {
  /**
   * Extracts a {@code FunctionSignature.WithValues<Object, SkylarkType>} from an annotation
   * @param name the name of the function
   * @param annotation the annotation
   * @param defaultValues an optional list of default values
   * @param paramDoc an optional list into which to store documentation strings
   * @param enforcedTypesList an optional list into which to store effective types to enforce
   */
  // NB: the two arguments paramDoc and enforcedTypesList are used to "return" extra values via
  // side-effects, and that's ugly
  // TODO(bazel-team): use AutoValue to declare a value type to use as return value?
  public static FunctionSignature.WithValues<Object, SkylarkType> getSignature(
      String name, SkylarkSignature annotation,
      @Nullable Iterable<Object> defaultValues,
      @Nullable List<String> paramDoc, @Nullable List<SkylarkType> enforcedTypesList) {

    Preconditions.checkArgument(name.equals(annotation.name()),
        "%s != %s", name, annotation.name());
    ArrayList<Parameter<Object, SkylarkType>> paramList = new ArrayList<>();
    HashMap<String, SkylarkType> enforcedTypes = enforcedTypesList == null
        ? null : new HashMap<String, SkylarkType>();

    HashMap<String, String> doc = new HashMap<>();
    boolean documented = annotation.documented();
    if (annotation.doc().isEmpty() && documented) {
      throw new RuntimeException(String.format("function %s is undocumented", name));
    }

    Iterator<Object> defaultValuesIterator = defaultValues == null
        ? null : defaultValues.iterator();
    try {
      for (Param param : annotation.mandatoryPositionals()) {
        paramList.add(getParameter(name, param, enforcedTypes, doc, documented,
                /*mandatory=*/true, /*star=*/false, /*starStar=*/false, /*defaultValue=*/null));
      }
      for (Param param : annotation.optionalPositionals()) {
        paramList.add(getParameter(name, param, enforcedTypes, doc, documented,
                /*mandatory=*/false, /*star=*/false, /*starStar=*/false,
                /*defaultValue=*/getDefaultValue(param, defaultValuesIterator)));
      }
      if (annotation.extraPositionals().length > 0
          || annotation.optionalNamedOnly().length > 0
          || annotation.mandatoryNamedOnly().length > 0) {
        @Nullable Param starParam = null;
        if (annotation.extraPositionals().length > 0) {
          Preconditions.checkArgument(annotation.extraPositionals().length == 1);
          starParam = annotation.extraPositionals()[0];
        }
        paramList.add(getParameter(name, starParam, enforcedTypes, doc, documented,
                /*mandatory=*/false, /*star=*/true, /*starStar=*/false, /*defaultValue=*/null));
      }
      for (Param param : annotation.mandatoryNamedOnly()) {
        paramList.add(getParameter(name, param, enforcedTypes, doc, documented,
                /*mandatory=*/true, /*star=*/false, /*starStar=*/false, /*defaultValue=*/null));
      }
      for (Param param : annotation.optionalNamedOnly()) {
        paramList.add(getParameter(name, param, enforcedTypes, doc, documented,
                /*mandatory=*/false, /*star=*/false, /*starStar=*/false,
                /*defaultValue=*/getDefaultValue(param, defaultValuesIterator)));
      }
      if (annotation.extraKeywords().length > 0) {
        Preconditions.checkArgument(annotation.extraKeywords().length == 1);
        paramList.add(
            getParameter(name, annotation.extraKeywords()[0], enforcedTypes, doc, documented,
                /*mandatory=*/false, /*star=*/false, /*starStar=*/true, /*defaultValue=*/null));
      }
      FunctionSignature.WithValues<Object, SkylarkType> signature =
          FunctionSignature.WithValues.<Object, SkylarkType>of(paramList);
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
      throw new RuntimeException(String.format("parameter %s is undocumented", name));
    }
    if (paramDoc != null) {
      paramDoc.put(param.name(), param.doc());
    }
    if (starStar) {
      return new Parameter.StarStar<>(param.name(), officialType);
    } else if (star) {
      return new Parameter.Star<>(param.name(), officialType);
    } else if (mandatory) {
      return new Parameter.Mandatory<>(param.name(), officialType);
    } else if (defaultValue != null && enforcedType != null) {
      Preconditions.checkArgument(enforcedType.contains(defaultValue),
          "In function '%s', parameter '%s' has default value %s that isn't of enforced type %s",
          name, param.name(), Printer.repr(defaultValue), enforcedType);
    }
    return new Parameter.Optional<>(param.name(), officialType, defaultValue);
  }

  private static Object getDefaultValue(Param param, Iterator<Object> iterator) {
    if (iterator != null) {
      return iterator.next();
    } else if (param.defaultValue().isEmpty()) {
      return Runtime.NONE;
    } else {
      try (Mutability mutability = Mutability.create("initialization")) {
        return Environment.builder(mutability)
            .setSkylark()
            .setGlobals(Environment.CONSTANTS_ONLY)
            .setEventHandler(Environment.FAIL_FAST_HANDLER)
            .build()
            .update("unbound", Runtime.UNBOUND)
            .eval(param.defaultValue());
      } catch (Exception e) {
        throw new RuntimeException(String.format(
            "Exception while processing @SkylarkSignature.Param %s, default value %s",
            param.name(), param.defaultValue()), e);
      }
    }
  }

  /** Extract additional signature information for BuiltinFunction-s */
  public static ExtraArgKind[] getExtraArgs(SkylarkSignature annotation) {
    final int numExtraArgs = (annotation.useLocation() ? 1 : 0)
        + (annotation.useAst() ? 1 : 0) + (annotation.useEnvironment() ? 1 : 0);
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
   * Configure all BaseFunction-s in a class from their @SkylarkSignature annotations
   * @param type a class containing BuiltinFunction fields that need be configured.
   * This function is typically called in a static block to initialize a class,
   * i.e. a class {@code Foo} containing @SkylarkSignature annotations would end with
   * {@code static { SkylarkSignatureProcessor.configureSkylarkFunctions(Foo.class); }}
   */
  public static void configureSkylarkFunctions(Class<?> type) {
    for (Field field : type.getDeclaredFields()) {
      if (field.isAnnotationPresent(SkylarkSignature.class)) {
        // The annotated fields are often private, but we need access them.
        field.setAccessible(true);
        SkylarkSignature annotation = field.getAnnotation(SkylarkSignature.class);
        Object value = null;
        try {
          value = Preconditions.checkNotNull(field.get(null),
              String.format(
                  "Error while trying to configure %s.%s: its value is null", type, field));
          if (BaseFunction.class.isAssignableFrom(field.getType())) {
            BaseFunction function = (BaseFunction) value;
            if (!function.isConfigured()) {
              function.configure(annotation);
            }
            Class<?> nameSpace = function.getObjectType();
            if (nameSpace != null) {
              Preconditions.checkState(!(function instanceof BuiltinFunction.Factory));
              nameSpace = Runtime.getCanonicalRepresentation(nameSpace);
              Runtime.registerFunction(nameSpace, function);
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
