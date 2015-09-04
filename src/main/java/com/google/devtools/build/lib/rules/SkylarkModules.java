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
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.SkylarkNativeModule;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Runtime;

/**
 * The basis for a Skylark Environment with all build-related modules registered.
 */
public final class SkylarkModules {

  private SkylarkModules() { }

  /**
   * The list of built in Skylark modules.
   * Documentation is generated automatically for all these modules.
   * They are also registered with the {@link Environment}.
   */
  public static final ImmutableList<Class<?>> MODULES = ImmutableList.of(
      SkylarkAttr.class,
      SkylarkCommandLine.class,
      SkylarkNativeModule.class,
      SkylarkRuleClassFunctions.class,
      SkylarkRuleImplementationFunctions.class);

  /** Global bindings for all Skylark modules */
  public static final Environment.Frame GLOBALS = createGlobals();

  private static Environment.Frame createGlobals() {
    try (Mutability mutability = Mutability.create("SkylarkModules")) {
      Environment env = Environment.builder(mutability)
          .setSkylark()
          .setGlobals(Environment.SKYLARK)
          .build();
      for (Class<?> moduleClass : MODULES) {
        Runtime.registerModuleGlobals(env, moduleClass);
      }
      return env.getGlobals();
    }
  }


  /**
   * Create an {@link Environment} in which to load a Skylark file.
   * @param eventHandler an EventHandler for warnings, errors, etc.
   * @param astFileContentHashCode a hash code identifying the file being evaluated
   * @param mutability the Mutability for the current evaluation context
   * @return a new Environment with the elements of the Skylark modules.
   */
  public static Environment getNewEnvironment(
      EventHandler eventHandler, String astFileContentHashCode, Mutability mutability) {
    return Environment.builder(mutability)
        .setSkylark()
        .setGlobals(GLOBALS)
        .setEventHandler(eventHandler)
        .setFileContentHashCode(astFileContentHashCode)
        .build();
  }

  @VisibleForTesting
  public static Environment getNewEnvironment(
      EventHandler eventHandler, Mutability mutability) {
    return getNewEnvironment(eventHandler, null, mutability);
  }
}
