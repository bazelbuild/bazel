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
 * Specifies that AutoCodec should generate a codec implementation for the annotated class. This is
 * generally only needed in the following cases:
 *
 * <ol>
 * <li>Interning work. {@link com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey}
 * </li>
 * <li>Non-trivial calculations and field initialization. {@link
 * com.google.devtools.build.lib.pkgcache.TestFilter} </li>
 * <li>Some paths are forbidden for DynamicCodec. {@link
 * com.google.devtools.build.lib.skyframe.serialization.AutoRegistry} </li>
 * </ol>
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
 */
@Target(ElementType.TYPE)
public @interface AutoCodec {
  // AutoCodec works by determining a unique *instantiator*, either a constructor or factory method,
  // to serve as a specification for serialization. The @AutoCodec.Instantiator tag can be helpful
  // for marking a specific instantiator.
  //
  // AutoCodec inspects the parameters of the instantiator and finds fields of the class
  // corresponding in both name and type. For serialization, it generates code that reads those
  // fields using reflection. For deserialization it generates code to invoke the instantiator.

  /**
   * Marks a specific method to use as the instantiator.
   *
   * <p>This marking is required when the class has more than one constructor.
   *
   * <p>Indicates an instantiator, either a constructor or factory method, for codec generation. A
   * compile-time error will result if multiple methods are thus tagged.
   */
  @Target({ElementType.CONSTRUCTOR, ElementType.METHOD})
  @interface Instantiator {}

  /**
   * Marks a static method to use for interning.
   *
   * <p>The method must accept an instance of the enclosing {@code AutoCodec} tagged class and
   * return an instance of the tagged class.
   */
  @Target({ElementType.METHOD})
  @interface Interner {}

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
}
