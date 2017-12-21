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
 * Specifies that AutoCodec should generate a codec implementation for the annotated class.
 *
 * <p>Example:
 *
 * <pre>{@code
 * @AutoCodec
 * class Target {
 *   public static final ObjectCodec<Target> CODEC = new Target_AutoCodec();
 * }
 * }</pre>
 *
 * The {@code _AutoCodec} suffix is added to the {@code Target} to obtain the generated class name.
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
     * Uses the first constructor of the class to synthesize a codec.
     *
     * <p>This strategy depends on
     *
     * <ul>
     *   <li>the first class constructor taking all serialized fields as parameters
     *   <li>and each serialized field having a corresponding getter.
     * </ul>
     *
     * For example, a constructor having parameter, {@code target}, should having a matching getter,
     * {@code getTarget()}.
     *
     * <p>The first constructor is the first ocurring in the source code.
     */
    CONSTRUCTOR,
    /**
     * Uses the public fields to infer serialization code.
     *
     * <p>Serializes each public field. Calls the no-arg constructor of the class to instantiate an
     * instance for deserialization.
     */
    PUBLIC_FIELDS,
    /**
     * For use with abstract classes (enforced at compile time).
     *
     * <p>Uses reflection to determine the concrete subclass, stores the name of the subclass and
     * uses its codec to serialize the data.
     */
    POLYMORPHIC,
  }

  Strategy strategy() default Strategy.CONSTRUCTOR;
}
