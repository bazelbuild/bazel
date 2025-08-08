// Copyright 2025 The Bazel Authors. All rights reserved.
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

package net.starlark.java.eval;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.math.BigInteger;
import java.util.HashSet;
import java.util.Objects;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;
import net.starlark.java.types.Types.DictType;
import net.starlark.java.types.Types.ListType;
import net.starlark.java.types.Types.SetType;
import net.starlark.java.types.Types.TupleType;
import net.starlark.java.types.Types.UnionType;

/** Type checker for Starlark types. */
public final class TypeChecker {

  private static boolean isTupleSubtypeOf(TupleType tuple1, TupleType tuple2) {
    if (tuple1.getElementTypes().size() != tuple2.getElementTypes().size()) {
      return false;
    }
    // Tuples are covariant
    for (int i = 0; i < tuple1.getElementTypes().size(); ++i) {
      if (!isSubtypeOf(tuple1.getElementTypes().get(i), tuple2.getElementTypes().get(i))) {
        return false;
      }
    }
    return true;
  }

  private static boolean isUnionSubtypeOf(
      ImmutableSet<StarlarkType> subtypes1, ImmutableSet<StarlarkType> subtypes2) {
    HashSet<StarlarkType> remainingSubtypes1 = new HashSet<>(subtypes1);
    // happy path - works only for exact matches
    // for example: Int < Int|Str, where Int is an exact match of Int
    remainingSubtypes1.removeAll(subtypes2);

    // This is the price we need to pay for having untagged unions
    // TODO(ilist@): Test this code path once we have collection types
    for (StarlarkType t2 : subtypes2) {
      // we need to call isSubtype, for example isSubtype(List[Int], List[Int | Str])
      for (StarlarkType t1 : ImmutableList.copyOf(remainingSubtypes1)) {
        if (isSubtypeOf(t1, t2)) {
          remainingSubtypes1.remove(t1);
        }
      }
    }
    return remainingSubtypes1.isEmpty();
  }

  public static boolean isSubtypeOf(StarlarkType type1, StarlarkType type2) {
    // Primitive unification, this way the lattice doesn't collapse
    if (Objects.equals(type1, Types.ANY)) {
      type1 = type2;
    } else if (Objects.equals(type2, Types.ANY)) {
      type2 = type1;
    }

    // TODO(ilist@): test this code path ("object" is not exposed to Starlark methods)
    if (type2.equals(Types.OBJECT)) {
      return true;
    }

    // normalize unions
    if (type1 instanceof UnionType union1) {
      if (type2 instanceof UnionType union2) {
        return isUnionSubtypeOf(union1.getTypes(), union2.getTypes());
      } else {
        return isUnionSubtypeOf(union1.getTypes(), ImmutableSet.of(type2)); // a|b < b
      }
    } else if (type2 instanceof UnionType union2) {
      return isUnionSubtypeOf(ImmutableSet.of(type1), union2.getTypes()); // a < a|b
    }

    // Mutable collections are invariant (which is necessary while the interface supports both
    // reading and modification). This matches Python's behaviour.
    if (type1 instanceof TupleType tuple1 && type2 instanceof TupleType tuple2) {
      return isTupleSubtypeOf(tuple1, tuple2);
    }
    if (type1 instanceof ListType list1 && type2 instanceof ListType list2) {
      return isEqual(list1.getElementType(), list2.getElementType());
    }
    if (type1 instanceof DictType dict1 && type2 instanceof DictType dict2) {
      return isEqual(dict1.getKeyType(), dict2.getKeyType())
          && isEqual(dict1.getValueType(), dict2.getValueType());
    }
    if (type1 instanceof SetType set1 && type2 instanceof SetType set2) {
      return isEqual(set1.getElementType(), set2.getElementType());
    }

    // TODO(ilist@): this just works for primitive types
    return Objects.equals(type1, type2);
  }

  private static boolean isEqual(StarlarkType type1, StarlarkType type2) {
    return isSubtypeOf(type1, type2) && isSubtypeOf(type2, type1);
  }

  static boolean isValueSubtypeOf(Object value, StarlarkType type2) {
    // Fast path for Any type. `type(value)` below can take long time to evaluate
    if (Objects.equals(type2, Types.ANY)) {
      return true;
    }
    return isSubtypeOf(type(value), type2);
  }

  static StarlarkType type(Object value) {
    if (value instanceof StarlarkValue val) {
      StarlarkType type = val.getStarlarkType();
      // Workaround for test mocks that generate getStarlarkType returning null
      return Objects.requireNonNullElse(type, Types.ANY);
    }
    if (value instanceof Boolean) {
      return Types.BOOL;
    }
    if (value instanceof String) {
      return Types.STR;
    }
    throw new IllegalArgumentException("Expected a valid Starlark value.");
  }

  public static StarlarkType fromJava(Class<?> cls) {
    StarlarkType t;
    if (cls == NoneType.class || cls == void.class) {
      t = Types.NONE;
    } else if (cls == String.class) {
      t = Types.STR;
    } else if (cls == Boolean.class || cls == boolean.class) {
      t = Types.BOOL;
    } else if (cls == int.class
        || cls == long.class
        || cls == Integer.class
        || cls == Long.class
        || cls == StarlarkInt.class
        || BigInteger.class.isAssignableFrom(cls)) {
      t = Types.INT;
    } else if (cls == double.class || cls == Double.class || cls == StarlarkFloat.class) {
      t = Types.FLOAT;
    } else if (cls == Object.class || cls == StarlarkValue.class) {
      return Types.OBJECT;
    } else {
      // TODO(ilist@): handle more complex types
      return Types.ANY;
    }
    return t;
  }

  private TypeChecker() {}
}
