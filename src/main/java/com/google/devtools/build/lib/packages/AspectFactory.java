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

package com.google.devtools.build.lib.packages;

/**
 * Creates the Skyframe node of an aspect.
 *
 * <p>Also has a reference to the definition of the aspect.
 */
public interface AspectFactory<TConfiguredTarget, TRuleContext, TAspect> {
  /**
   * Creates the aspect based on the configured target of the associated rule.
   *
   * @param base the configured target of the associated rule
   * @param context the context of the associated configured target plus all the attributes the
   *     aspect itself has defined
   * @param parameters information from attributes of the rule that have requested this
   *     aspect
   */
  TAspect create(TConfiguredTarget base, TRuleContext context, AspectParameters parameters)
      throws InterruptedException;

  /**
   * Returns the definition of the aspect.
   */
  AspectDefinition getDefinition();

  /**
   * Dummy wrapper class for utility methods because interfaces cannot even have static ones.
   */
  public static final class Util {
    private Util() {
      // Should never be instantiated
    }

    public static AspectFactory<?, ?, ?> create(AspectClass aspectClass) {
      // TODO(bazel-team): This should be cached somehow, because this method is invoked quite often

      return aspectClass.newInstance();
    }
  }
}
