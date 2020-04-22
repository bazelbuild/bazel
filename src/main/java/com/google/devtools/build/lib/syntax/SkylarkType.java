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
package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import javax.annotation.Nullable;

/** A SkylarkType represents the top-level class of the elements of a Depset. */
// TODO(adonovan): move to Depset.ElementType.
// TODO(adonovan): consider deleting this class entirely and using Class directly
// within Depset. Depset.getContentType would need to document "null means empty",
// but almost every caller just wants to stringify it.
@Immutable
public final class SkylarkType {

  @Nullable private final Class<?> cls; // null => empty depset

  private SkylarkType(@Nullable Class<?> cls) {
    this.cls = cls;
  }

  /** The element type of the empty depset. */
  public static final SkylarkType EMPTY = new SkylarkType(null);

  /** The element type of a depset of strings. */
  public static final SkylarkType STRING = of(String.class);

  @Override
  public String toString() {
    return cls == null ? "empty" : EvalUtils.getDataTypeNameFromClass(cls);
  }

  /** Returns the symbol representing the element type of a depset. */
  public static SkylarkType of(Class<?> cls) {
    return new SkylarkType(getTypeClass(cls));
  }

  // Returns the Java class representing the Starlark type of an instance of cls, which must be one
  // of String, Integer, or Boolean (in which case the result is cls), or a SkylarkModule-annotated
  // Starlark value class or one of its subclasses, in which case the result is the annotated class.
  //
  // TODO(adonovan): consider publishing something like this as Starlark.typeClass when we clean up
  // the various EvalUtils.getDataType operators.
  private static Class<?> getTypeClass(Class<?> cls) {
    if (cls == String.class || cls == Integer.class || cls == Boolean.class) {
      return cls; // fast path for common case
    }
    Class<?> superclass = SkylarkInterfaceUtils.getParentWithSkylarkModule(cls);
    if (superclass != null) {
      return superclass;
    }
    if (!StarlarkValue.class.isAssignableFrom(cls)) {
      throw new IllegalArgumentException(
          "invalid Depset element type: " + cls.getName() + " is not a subclass of StarlarkValue");
    }
    return cls;
  }

  /** Returns the symbol representing elements of the same class as x. */
  // Called by Depset element insertion.
  static SkylarkType ofValue(Object x) {
    return of(x.getClass());
  }

  // Called by precondition check of Depset.getSet conversion.
  //
  // Fails if cls is neither Object.class nor a valid Starlark value class. One might expect that if
  // a SkylarkType canBeCastTo Integer, then it can also be cast to Number, but this is not the
  // case: getTypeClass fails if passed a supertype of a Starlark class that is not itself a valid
  // Starlark value class. As a special case, Object.class is permitted, and represents "any value".
  //
  // This leads one to wonder why canBeCastTo calls getTypeClass at all. The answer is that it is
  // yet another hack to support skylarkbuildapi. For example, (FileApi).canBeCastTo(Artifact.class)
  // reports true, because a Depset whose elements are nominally of type FileApi is assumed to
  // actually contain only elements of class Artifact. If there were a second implementation of
  // FileAPI, the operation would be unsafe.
  //
  // TODO(adonovan): once skylarkbuildapi has been deleted, eliminate the getTypeClass calls here
  // and in SkylarkType.of, and remove the special case for Object.class since isAssignableFrom will
  // allow any supertype of the element type, whether or not it is a Starlark value class.
  boolean canBeCastTo(Class<?> cls) {
    return this.cls == null
        || cls == Object.class // historical exception
        || getTypeClass(cls).isAssignableFrom(this.cls);
  }

  @Override
  public int hashCode() {
    return cls == null ? 0 : cls.hashCode();
  }

  @Override
  public boolean equals(Object that) {
    return that instanceof SkylarkType && this.cls == ((SkylarkType) that).cls;
  }
}
