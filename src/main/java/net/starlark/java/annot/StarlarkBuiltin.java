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
package net.starlark.java.annot;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * This annotation is used on classes and interfaces that represent Starlark data types.
 *
 * <p>Conceptually, every {@link StarlarkBuiltin} annotation corresponds to a user-distinguishable
 * Starlark type. The annotation holds metadata associated with that type, in particular its name
 * and documentation. The annotation also implicitly demarcates the Starlark API of the type. It
 * does not matter whether the annotation is used on a class or an interface.
 *
 * <p>Annotations are "inherited" and "overridden", in the sense that a child class or interface
 * takes on the Starlark type of its ancestor by default, unless it has a direct annotation of its
 * own. If there are multiple ancestors that have an annotation, then to avoid ambiguity we require
 * that one of them is a subtype of the rest; that is the one whose annotation gets inherited. This
 * ensures that every class implements at most one Starlark type, and not an ad hoc hybrid of
 * multiple types. (In mathematical terms, the most-derived annotation for class or interface C is
 * the minimum element in the partial order of all annotations defined on C and its ancestors, where
 * the order relationship is X < Y if X annotates a subtype of what Y annotates.) The lookup logic
 * for retrieving a class's {@link StarlarkBuiltin} is implemented by {@link
 * StarlarkInterfaceUtils#getStarlarkBuiltin}.
 *
 * <p>Inheriting an annotation is useful when the class is an implementation detail, such as a
 * concrete implementation of an abstract interface. Overriding an annotation is useful when the
 * class should have its own distinct user-visible API or documentation. For example, {@link
 * Sequence} is an abstract type implemented by both {@link StarlarkList} and {@link
 * Sequence.Tuple}, all three of which are annotated. Annotating the list and tuple types allows
 * them to define different methods, while annotating {@link Sequence} allows them to be identified
 * as a single type for the purpose of type checking, documentation, and error messages.
 *
 * <p>All {@link StarlarkBuiltin}-annotated types must implement {@link StarlarkValue}. Nearly all
 * non-abstract implementations of {@link StarlarkValue} have or inherit a {@link StarlarkBuiltin}
 * annotation. (It is possible, though quite unusual, to declare an implementation of {@code
 * StarlarkValue} without using the annotation mechanism defined in this package. {@code
 * StarlarkFunction} is one example.)
 */
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface StarlarkBuiltin {

  /** A type name that may be used in stringification and error messages. */
  String name();

  /** A title for the documentation page generated for this type. */
  String title() default "";

  /** Module documentation in HTML. May be empty only if {@code !documented()}. */
  String doc() default "";

  /** Whether the module should appear in the documentation. */
  boolean documented() default true;

  StarlarkDocumentationCategory category() default StarlarkDocumentationCategory.TOP_LEVEL_TYPE;
}
