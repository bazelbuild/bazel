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

package com.google.devtools.common.options;

import com.google.common.collect.ImmutableList;
import java.lang.annotation.ElementType;
import java.lang.annotation.Inherited;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.time.Duration;
import java.util.List;

/**
 * Applied to an {@link OptionsBase} subclass to indicate that all of its options fields have types
 * chosen from {@link #coreTypes}. Any subclasses of the class to which it's applied must also
 * satisfy the same property.
 *
 * <p>Options classes with this annotation are serializable and deeply immutable, except that the
 * fields of the options class can be reassigned (although this is bad practice).
 *
 * <p>Note that {@link Option#allowMultiple} is not allowed for options in classes with this
 * annotation, since their type is {@link List}.
 */
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Inherited
public @interface UsesOnlyCoreTypes {

  /**
   * These are the core options field types. They all have default converters, are deeply immutable,
   * and are serializable.
   *
   * Lists are not considered core types, so {@link Option#allowMultiple} options are not permitted.
   */
  public static final ImmutableList<Class<?>> CORE_TYPES = ImmutableList.of(
      // 1:1 correspondence with Converters.DEFAULT_CONVERTERS.
      String.class,
      int.class,
      long.class,
      double.class,
      boolean.class,
      TriState.class,
      Void.class,
      Duration.class
  );
}
