// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import java.lang.annotation.ElementType;
import java.lang.annotation.Target;

/**
 * Specifies that AutoCodec should generate a codec implementation for the annotated abstract class.
 *
 * <p>Example:
 *
 * <pre>{@code
 * @AutoCodec
 * abstract class Codec implements ObjectCodec<Target> {
 *   static Codec create() {
 *     return new AutoCodec_Target();
 *   }
 * }
 * }</pre>
 *
 * The {@code AutoCodec_} prefix is added to the {@Target} to obtain the generated class name.
 */
@Target(ElementType.TYPE)
public @interface AutoCodec {
  /**
   * AutoCodec recursively derives a codec using the public interfaces of the class.
   *
   * <p>Specific strategies are described below.
   */
  public static enum Strategy {
    /**
     * Uses the constructor to infer serialization code.
     *
     * <p>Each constructor parameter is expected to have a corresponding getter. These pairs are
     * used for serialization and deserialization.
     */
    CONSTRUCTOR,
    // TODO(shahan): Add a strategy that serializes from public members.
  }

  Strategy strategy() default Strategy.CONSTRUCTOR;
}
