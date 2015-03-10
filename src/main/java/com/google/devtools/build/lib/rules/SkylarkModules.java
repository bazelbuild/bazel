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
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.MethodLibrary;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkFunction;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.SkylarkType.SkylarkFunctionType;
import com.google.devtools.build.lib.syntax.ValidationEnvironment;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

/**
 * A class to handle all Skylark modules, to create and setup Validation and regular Environments.
 */
// TODO(bazel-team): move that to syntax/ and
// let each extension register itself in a static { } statement.
public class SkylarkModules {

  public static final ImmutableList<Class<?>> MODULES = ImmutableList.of(
      SkylarkAttr.class,
      SkylarkCommandLine.class,
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
    return getValidationEnvironment(ImmutableMap.<String, SkylarkType>of());
  }

  /**
   * Returns a new ValidationEnvironment with the elements of the Skylark modules and extraObjects.
   */
  public static ValidationEnvironment getValidationEnvironment(
      ImmutableMap<String, SkylarkType> extraObjects) {
    Map<SkylarkType, Map<String, SkylarkType>> builtIn = new HashMap<>();
    Map<String, SkylarkType> global = new HashMap<>();
    builtIn.put(SkylarkType.GLOBAL, global);
    collectSkylarkTypesFromFields(Environment.class, builtIn);
    for (Class<?> moduleClass : MODULES) {
      if (moduleClass.isAnnotationPresent(SkylarkModule.class)) {
        global.put(moduleClass.getAnnotation(SkylarkModule.class).name(),
            SkylarkType.of(moduleClass));
      }
    }
    global.put("native", SkylarkType.UNKNOWN);
    MethodLibrary.setupValidationEnvironment(builtIn);
    for (Class<?> module : MODULES) {
      collectSkylarkTypesFromFields(module, builtIn);
    }
    global.putAll(extraObjects);
    return new ValidationEnvironment(CollectionUtils.toImmutable(builtIn));
  }

  /**
   * Collects the SkylarkFunctions from the fields of the class of the object parameter
   * and adds them into the builder.
   */
  private static void collectSkylarkFunctionsAndObjectsFromFields(Class<?> type,
      ImmutableList.Builder<Function> functions, ImmutableMap.Builder<String, Object> objects) {
    try {
      for (Field field : type.getDeclaredFields()) {
        if (field.isAnnotationPresent(SkylarkBuiltin.class)) {
          // Fields in Skylark modules are sometimes private. Nevertheless they have to
          // be annotated with SkylarkBuiltin.
          field.setAccessible(true);
          SkylarkBuiltin annotation = field.getAnnotation(SkylarkBuiltin.class);
          if (SkylarkFunction.class.isAssignableFrom(field.getType())) {
            SkylarkFunction function = (SkylarkFunction) field.get(null);
            if (!function.isConfigured()) {
              function.configure(annotation);
            }
            functions.add(function);
          } else {
            objects.put(annotation.name(), field.get(null));
          }
        }
      }
    } catch (IllegalArgumentException | IllegalAccessException e) {
      // This should never happen.
      throw new RuntimeException(e);
    }
  }

  /**
   * Collects the SkylarkFunctions from the fields of the class of the object parameter
   * and adds their class and their corresponding return value to the builder.
   */
  private static void collectSkylarkTypesFromFields(Class<?> classObject,
      Map<SkylarkType, Map<String, SkylarkType>> builtIn) {
    for (Field field : classObject.getDeclaredFields()) {
      if (field.isAnnotationPresent(SkylarkBuiltin.class)) {
        SkylarkBuiltin annotation = field.getAnnotation(SkylarkBuiltin.class);
        if (SkylarkFunction.class.isAssignableFrom(field.getType())) {
          try {
            // TODO(bazel-team): infer the correct types.
            SkylarkType objectType = annotation.objectType().equals(Object.class)
                ? SkylarkType.GLOBAL
                : SkylarkType.of(annotation.objectType());
            if (!builtIn.containsKey(objectType)) {
              builtIn.put(objectType, new HashMap<String, SkylarkType>());
            }
            // TODO(bazel-team): add parameters to SkylarkFunctionType
            SkylarkType returnType = SkylarkType.getReturnType(annotation);
            builtIn.get(objectType).put(annotation.name(),
                SkylarkFunctionType.of(annotation.name(), returnType));
          } catch (IllegalArgumentException e) {
            // This should never happen.
            throw new RuntimeException(e);
          }
        } else if (Function.class.isAssignableFrom(field.getType())) {
          builtIn.get(SkylarkType.GLOBAL).put(annotation.name(),
              SkylarkFunctionType.of(annotation.name(), SkylarkType.UNKNOWN));
        } else {
          builtIn.get(SkylarkType.GLOBAL).put(annotation.name(), SkylarkType.of(field.getType()));
        }
      }
    }
  }
}
