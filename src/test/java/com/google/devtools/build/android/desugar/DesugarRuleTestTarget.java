/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar;

/**
 * The source class used as data in {@link DesugarRuleTest}.
 *
 * <p>Note: This class is used for checking the desugar pipeline and {@link DesugarRule} working as
 * expected. DO NOT use this target for testing individual desugar logic.
 */
@SuppressWarnings({"PrivateConstructorForUtilityClass", "InterfaceWithOnlyStatics"}) // testing-only
class DesugarRuleTestTarget {
  interface InterfaceSubjectToDesugar {
    static void staticMethod() {}

    default int defaultMethod(int x, int y) {
      return x + y;
    }
  }
}
