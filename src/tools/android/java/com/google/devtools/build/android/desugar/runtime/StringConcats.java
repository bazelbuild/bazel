// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.runtime;

/** The runtime library used for string concatenation. */
public class StringConcats {

  /**
   * Concatenate strings in a way compliant with the Javadoc for {@link
   * java.lang.invoke.StringConcatFactory#makeConcatWithConstants}.
   */
  public static String concat(Object[] runtimeTexts, String recipe, Object[] constants) {
    StringBuilder stringBuilder = new StringBuilder();
    for (int i = 0, runTimeTextIndex = 0, constantIndex = 0; i < recipe.length(); i++) {
      char c = recipe.charAt(i);
      if (c == '\1') {
        stringBuilder.append(runtimeTexts[runTimeTextIndex++]);
      } else if (c == '\2') {
        stringBuilder.append(constants[constantIndex++]);
      } else {
        stringBuilder.append(c);
      }
    }
    return stringBuilder.toString();
  }

  private StringConcats() {}
}
