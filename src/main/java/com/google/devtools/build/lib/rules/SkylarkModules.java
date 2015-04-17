// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.rules;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.MethodLibrary;
import com.google.devtools.build.lib.packages.SkylarkNativeModule;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvaluationContext;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.ValidationEnvironment;

import java.lang.reflect.Field;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A class to handle all Skylark modules, to create and setup Validation and regular Environments.
 */
// TODO(bazel-team): move that to the syntax package and
// let each extension register itself in a static { } statement.
public class SkylarkModules {

  /**
   * The list of built in Skylark modules. Documentation is generated automatically for all these
   * modules. They are also registered with the {@link ValidationEnvironment} and the
   * {@link SkylarkEnvironment}. Note that only {@link SkylarkFunction}s are handled properly.
   */
  // TODO(bazel-team): find a more general, more automated way of registering classes and building
  // initial environments. And don't give syntax.Environment and packages.MethodLibrary a special
  // treatment, have them use the same registration mechanism as other classes currently below.
  public static final ImmutableList<Class<?>> MODULES = ImmutableList.of(
      SkylarkAttr.class,
      SkylarkCommandLine.class,
      SkylarkNativeModule.class,
      SkylarkRuleClassFunctions.class,
      SkylarkRuleImplementationFunctions.class);

  private static final ImmutableMap<Class<?>, ImmutableList<Function>> FUNCTION_MAP;
  private static final ImmutableMap<String, Object> OBJECTS;

  static {
    try {
      ImmutableMap.Builder<Class<?>, ImmutableList<Function>> functionMap = ImmutableMap.builder();
      ImmutableMap.Builder<String, Object> objects = ImmutableMap.builder();
      for (Class<?> moduleClass : MODULES) {
        if (moduleClass.isAnnotationPresent(SkylarkModule.class)) {
          objects.put(moduleClass.getAnnotation(SkylarkModule.class).name(),
              moduleClass.newInstance());
        }
        ImmutableList.Builder<Function> functions = ImmutableList.builder();
        collectSkylarkFunctionsAndObjectsFromFields(moduleClass, functions, objects);
        functionMap.put(moduleClass, functions.build());
      }
      FUNCTION_MAP = functionMap.build();
      OBJECTS = objects.build();
    } catch (InstantiationException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Returns a new SkylarkEnvironment with the elements of the Skylark modules.
   */
  public static SkylarkEnvironment getNewEnvironment(
      EventHandler eventHandler, String astFileContentHashCode) {
    SkylarkEnvironment env = new SkylarkEnvironment(eventHandler, astFileContentHashCode);
    setupEnvironment(env);
    return env;
  }

  @VisibleForTesting
  public static SkylarkEnvironment getNewEnvironment(EventHandler eventHandler) {
    return getNewEnvironment(eventHandler, null);
  }

  private static void setupEnvironment(Environment env) {
    MethodLibrary.setupMethodEnvironment(env);
    for (Map.Entry<Class<?>, ImmutableList<Function>> entry : FUNCTION_MAP.entrySet()) {
      for (Function function : entry.getValue()) {
        if (function.getObjectType() != null) {
          env.registerFunction(function.getObjectType(), function.getName(), function);
        } else {
          env.update(function.getName(), function);
        }
      }
    }
    for (Map.Entry<String, Object> entry : OBJECTS.entrySet()) {
      env.update(entry.getKey(), entry.getValue());
    }
  }

  /**
   * Returns a new ValidationEnvironment with the elements of the Skylark modules.
   */
  public static ValidationEnvironment getValidationEnvironment() {
    return getValidationEnvironment(ImmutableSet.<String>of());
  }

  /**
   * Returns a new ValidationEnvironment with the elements of the Skylark modules and extraObjects.
   */
  public static ValidationEnvironment getValidationEnvironment(ImmutableSet<String> extraObjects) {
    Set<String> builtIn = new HashSet<>();
    collectSkylarkTypesFromFields(Environment.class, builtIn);
    for (Class<?> moduleClass : MODULES) {
      if (moduleClass.isAnnotationPresent(SkylarkModule.class)) {
        builtIn.add(moduleClass.getAnnotation(SkylarkModule.class).name());
      }
    }
    MethodLibrary.setupValidationEnvironment(builtIn);
    for (Class<?> module : MODULES) {
      collectSkylarkTypesFromFields(module, builtIn);
    }
    builtIn.addAll(extraObjects);
    return new ValidationEnvironment(builtIn);
  }

  public static EvaluationContext newEvaluationContext(EventHandler eventHandler) {
    return EvaluationContext.newSkylarkContext(
        getNewEnvironment(eventHandler), getValidationEnvironment());
  }

  /**
   * Collects the Functions from the fields of the class of the object parameter
   * and adds them into the builder.
   */
  private static void collectSkylarkFunctionsAndObjectsFromFields(Class<?> type,
      ImmutableList.Builder<Function> functions, ImmutableMap.Builder<String, Object> objects) {
    try {
      for (Field field : type.getDeclaredFields()) {
        if (field.isAnnotationPresent(SkylarkBuiltin.class)) {
          // Fields in Skylark modules are sometimes private.
          // Nevertheless they have to be annotated with SkylarkBuiltin.
          field.setAccessible(true);
          SkylarkBuiltin annotation = field.getAnnotation(SkylarkBuiltin.class);
          Object value = field.get(null);
          if (SkylarkFunction.class.isAssignableFrom(field.getType())) {
            SkylarkFunction function = (SkylarkFunction) value;
            if (!function.isConfigured()) {
              function.configure(annotation);
            }
            functions.add(function);
          } else {
            objects.put(annotation.name(), value);
          }
        }
      }
    } catch (IllegalArgumentException | IllegalAccessException e) {
      // This should never happen.
      throw new RuntimeException(e);
    }
  }

  /**
   * Collects the Functions from the fields of the class of the object parameter
   * and adds their class and their corresponding return value to the builder.
   */
  private static void collectSkylarkTypesFromFields(Class<?> classObject, Set<String> builtIn) {
    for (Field field : classObject.getDeclaredFields()) {
      if (field.isAnnotationPresent(SkylarkBuiltin.class)) {
        SkylarkBuiltin annotation = field.getAnnotation(SkylarkBuiltin.class);
        if (SkylarkFunction.class.isAssignableFrom(field.getType())) {
          // Ignore non-global values.
          if (annotation.objectType().equals(Object.class)) {
            builtIn.add(annotation.name());
          }
        } else {
          builtIn.add(annotation.name());
        }
      }
    }
  }
}
