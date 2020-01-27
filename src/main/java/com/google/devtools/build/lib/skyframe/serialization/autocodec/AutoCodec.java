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
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Specifies that AutoCodec should generate a codec implementation for the annotated class. For
 * classes, this is generally only needed if they do interning or other non-trivial creation-time
 * work.
 *
 * <p>Example:
 *
 * <pre>{@code
 * @AutoCodec
 * class Target {
 * }</pre>
 *
 * The {@code _AutoCodec} suffix is added to the {@code Target} to obtain the generated class name.
 * In the example, that results in a class named {@code Target_AutoCodec} but applications should
 * not need to directly access the generated class.
 *
 * <p>If applied to a field (which must be static and final), the field is stored as a "constant"
 * allowing for trivial serialization of it as an integer tag (see {@code CodecScanner} and
 * {@code ObjectCodecRegistry}). In order to do that, a trivial associated "RegisteredSingleton"
 * class is generated. Tagging such a field is harmless, and can be done conservatively.
 */
@Target({ElementType.TYPE, ElementType.FIELD})
// TODO(janakr): remove once serialization is complete.
@Retention(RetentionPolicy.RUNTIME)
// TODO(janakr): remove unnecessary @AutoCodec annotations throughout our codebase.
public @interface AutoCodec {
  /**
   * AutoCodec recursively derives a codec using the public interfaces of the class.
   *
   * <p>Specific strategies are described below.
   */
  enum Strategy {
    /**
     * Uses a constructor or factory method of the class to synthesize a codec.
     *
     * <p>This strategy depends on
     *
     * <ul>
     *   <li>a designated constructor or factory method to inspect to generate the codec
     *   <li>each parameter must match a member field on name and the field will be interpreted as
     *       an instance of the parameter type.
     * </ul>
     *
     * <p>If there is a unique constructor, @AutoCodec may select that as the default instantiator,
     * otherwise one must be selected using the {@link AutoCodec.Instantiator} annotation.
     */
    INSTANTIATOR,
    /**
     * For use with {@link com.google.auto.value.AutoValue} classes with an {@link
     * com.google.auto.value.AutoValue.Builder} static nested Builder class: uses the builder when
     * deserializing.
     */
    AUTO_VALUE_BUILDER,
  }

  /**
   * Marks a specific method when using the INSTANTIATOR strategy.
   *
   * <p>Indicates an instantiator, either a constructor or factory method, for codec generation. A
   * compile-time error will result if multiple methods are thus tagged.
   */
  @Target({ElementType.CONSTRUCTOR, ElementType.METHOD})
  @interface Instantiator {}

  Strategy strategy() default Strategy.INSTANTIATOR;

  /**
   * Checks whether or not this class is allowed to be serialized. See {@link
   * com.google.devtools.build.lib.skyframe.serialization.SerializationContext#checkClassExplicitlyAllowed}.
   */
  boolean checkClassExplicitlyAllowed() default false;

  /**
   * Adds an explicitly allowed class for this serialization session. See {@link
   * com.google.devtools.build.lib.skyframe.serialization.SerializationContext#addExplicitlyAllowedClass}.
   */
  Class<?>[] explicitlyAllowClass() default {};

  /**
   * Signals that the annotated element is only visible for use by serialization. It should not be
   * used by other callers.
   *
   * <p>TODO(janakr): Add an ErrorProne checker to enforce this.
   */
  @Target({ElementType.TYPE, ElementType.METHOD, ElementType.CONSTRUCTOR, ElementType.FIELD})
  @interface VisibleForSerialization {}
}
