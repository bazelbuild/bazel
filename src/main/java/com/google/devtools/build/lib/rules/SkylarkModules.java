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
import com.google.devtools.build.lib.syntax.EvaluationContext;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.ValidationEnvironment;

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
    for (Class<?> moduleClass : MODULES) {
      Runtime.registerModuleGlobals(env, moduleClass);
    }
    // Even though PACKAGE_NAME has no value _now_ and will be bound later,
    // it needs to be visible for the ValidationEnvironment to be happy.
    env.update(Runtime.PKG_NAME, Runtime.NONE);
  }

  /**
   * Returns a new ValidationEnvironment with the elements of the Skylark modules.
   */
  public static ValidationEnvironment getValidationEnvironment() {
    // TODO(bazel-team): refactor constructors so we don't have those null-s
    return new ValidationEnvironment(getNewEnvironment(null));
  }

  public static EvaluationContext newEvaluationContext(EventHandler eventHandler) {
    return EvaluationContext.newSkylarkContext(
        getNewEnvironment(eventHandler), getValidationEnvironment());
  }
}
