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
package com.google.devtools.build.lib.skylarkinterface;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * This annotation is used on classes and interfaces that represent Skylark data types.
 *
 * <p>Conceptually, every {@code @SkylarkModule} annotation corresponds to a user-distinguishable
 * Skylark type. The annotation holds metadata associated with that type, in particular its name and
 * documentation. The annotation also implicitly demarcates the Skylark API of the type. It does not
 * matter whether the annotation is used on a class or an interface.
 *
 * <p>Annotations are "inherited" and "overridden", in the sense that a child class or interface
 * takes on the Skylark type of its ancestor by default, unless it has a direct annotation of its
 * own. If there are multiple ancestors that have an annotation, then to avoid ambiguity we require
 * that one of them is a subtype of the rest; that is the one whose annotation gets inherited. This
 * ensures that every class implements at most one Skylark type, and not an ad hoc hybrid of
 * multiple types. (In mathematical terms, the most-derived annotation for class or interface C is
 * the minimum element in the partial order of all annotations defined on C and its ancestors, where
 * the order relationship is X < Y if X annotates a subtype of what Y annotates.) The lookup logic
 * for retrieving a class's {@code @SkylarkModule} is implemented by {@link
 * SkylarkInterfaceUtils#getSkylarkModule}.
 *
 * <p>Inheriting an annotation is useful when the class is an implementation detail, such as a
 * concrete implementation of an abstract interface. Overriding an annotation is useful when the
 * class should have its own distinct user-visible API or documentation. For example, {@link
 * SkylarkList} is an abstract type implemented by both {@link SkylarkList.MutableList} and {@link
 * SkylarkList.Tuple}, all three of which are annotated. Annotating the list and tuple types allows
 * them to define different methods, while annotating {@link SkylarkList} allows them to be
 * identified as a single type for the purpose of type checking, documentation, and error messages.
 *
 * <p>All {@code @SkylarkModule}-annotated types should implement {@link SkylarkValue}. Conversely,
 * all non-abstract implementations of {@link SkylarkValue} should have or inherit a {@code
 * @SkylarkModule} annotation.
 */
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface SkylarkModule {

  /** A type name that may be used in stringification and error messages. */
  String name();

  /** A title for the documentation page generated for this type. */
  String title() default "";

  String doc();

  boolean documented() default true;

  /**
   * If true, this type is a singleton top-level type whose main purpose is to act as a namespace
   * for other values.
   */
  boolean namespace() default false;

  SkylarkModuleCategory category() default SkylarkModuleCategory.TOP_LEVEL_TYPE;
}
