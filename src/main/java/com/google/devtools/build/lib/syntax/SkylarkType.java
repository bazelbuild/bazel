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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A class representing types available in Skylark.
 *
 * <p>A SkylarkType can be one of:
 *
 * <ul>
 *   <li>a Simple type that contains exactly the objects in a given class, (including the special
 *       TOP and BOTTOM types that respectively contain all the objects (Simple type for
 *       Object.class) and no object at all (Simple type for EmptyType.class, isomorphic to
 *       Void.class).
 *   <li>a Combination of a generic class (one of SET, selector) and an argument type (that itself
 *       need not be Simple).
 *   <li>a Union of a finite set of types
 *   <li>a FunctionType associated with a name and a returnType
 * </ul>
 *
 * <p>In a style reminiscent of Java's null, Skylark's None is in all the types as far as type
 * inference goes, yet actually no type .contains(it).
 *
 * <p>The current implementation fails to distinguish between TOP and ANY, between BOTTOM and EMPTY
 * (VOID, ZERO, FALSE):
 *
 * <ul>
 *   <li>In type analysis, we often distinguish a notion of "the type of this object" from the
 *       notion of "what I know about the type of this object". Some languages have a Universal Base
 *       Class that contains all objects, and would be the ANY type. The Skylark runtime, written in
 *       Java, has this ANY type, Java's Object.class. But the Skylark validation engine doesn't
 *       really have a concept of an ANY class; however, it does have a concept of a yet-undermined
 *       class, the TOP class (called UNKNOWN in previous code). In the future, we may have to
 *       distinguish between the two, at which point type constructor classes would have to be
 *       generic in "actual type" vs "partial knowledge of type".
 *   <li>Similarly, and EMPTY type (also known as VOID, ZERO or FALSE, in other contexts) is a type
 *       that has no instance, whereas the BOTTOM type is the type analysis that says that there is
 *       no possible runtime type for the given object, which may imply that the point in the
 *       program at which the object is evaluated cannot be reached, etc.
 * </ul>
 *
 * So for now, we have puns between TOP and ANY, BOTTOM and EMPTY, between runtime (eval) and
 * validation-time (validate). Yet in the future, we may need to make a clear distinction,
 * especially if we are to have types such List(Any) vs List(Top), which contains the former, but
 * also plenty of other quite distinct types. And yet in a future future, the TOP type would not be
 * represented explicitly, instead a new type variable would be inserted everywhere a type is
 * unknown, to be unified with further type information as it becomes available.
 */
// TODO(bazel-team): move the FunctionType side-effect out of the type object
// and into the validation environment.
public abstract class SkylarkType {

  // The main primitives to override in subclasses

  /** Is the given value an element of this type? By default, no (empty type) */
  boolean contains(Object value) {
    return false;
  }

  /**
   * intersectWith() is the internal method from which function intersection(t1, t2) is computed.
   * OVERRIDE this method in your classes, but DO NOT TO CALL it: only call intersection(). When
   * computing intersection(t1, t2), whichever type defined before the other knows nothing about the
   * other and about their intersection, and returns BOTTOM; the other knows about the former, and
   * returns their intersection (which may be BOTTOM). intersection() will call in one order then
   * the other, and return whichever answer isn't BOTTOM, if any. By default, types are disjoint and
   * their intersection is BOTTOM.
   */
  // TODO(bazel-team): should we define and use an Exception instead?
  SkylarkType intersectWith(SkylarkType other) {
    return BOTTOM;
  }

  /** @return true if any object of this SkylarkType can be cast to that Java class */
  boolean canBeCastTo(Class<?> type) {
    return SkylarkType.of(type).includes(this);
  }

  /** @return the smallest java Class known to contain all elements of this type */
  // Note: most user-code should be using a variant that throws an Exception
  // if the result is Object.class but the type isn't TOP.
  Class<?> getType() {
    return Object.class;
  }

  static SkylarkType intersection(SkylarkType t1, SkylarkType t2) {
    if (t1.equals(t2)) {
      return t1;
    }
    SkylarkType t = t1.intersectWith(t2);
    if (t == BOTTOM) {
      return t2.intersectWith(t1);
    } else {
      return t;
    }
  }

  boolean includes(SkylarkType other) {
    return intersection(this, other).equals(other);
  }

  SkylarkType getArgType() {
    return TOP;
  }

  private static final class Empty {} // Empty type, used as basis for Bottom

  // Notable types

  /** A singleton for the TOP type, that at analysis time means that any type is possible. */
  @AutoCodec static final Simple TOP = new Top();

  /** A singleton for the BOTTOM type, that contains no element */
  @AutoCodec static final Simple BOTTOM = new Bottom();

  /** NONE, the Unit type, isomorphic to Void, except its unique element prints as None */
  // Note that we currently consider at validation time that None is in every type,
  // by declaring its type as TOP instead of NONE, even though at runtime,
  // we reject None from all types but NONE, and in particular from e.g. lists of Files.
  // TODO(bazel-team): resolve this inconsistency, one way or the other.
  @AutoCodec public static final Simple NONE = Simple.forClass(NoneType.class);

  /** The STRING type, for strings */
  @AutoCodec public static final Simple STRING = Simple.forClass(String.class);

  /** The INTEGER type, for 32-bit signed integers */
  @AutoCodec public static final Simple INT = Simple.forClass(Integer.class);

  /** The BOOLEAN type, that contains TRUE and FALSE */
  @AutoCodec public static final Simple BOOL = Simple.forClass(Boolean.class);

  /** The DICT type, that contains Dict */
  @AutoCodec static final Simple DICT = Simple.forClass(Dict.class);

  /** The LIST type, that contains all StarlarkList-s */
  @AutoCodec static final Simple LIST = Simple.forClass(StarlarkList.class);

  /** The TUPLE type, that contains all Tuple-s */
  @AutoCodec static final Simple TUPLE = Simple.forClass(Tuple.class);

  /** The SET type, that contains all Depsets, and the generic combinator for them */
  @AutoCodec static final Simple SET = Simple.forClass(Depset.class);

  // Common subclasses of SkylarkType

  /** the Top type contains all objects */
  private static class Top extends Simple {

    private Top() {
      super(Object.class);
    }

    @Override
    boolean contains(Object value) {
      return true;
    }

    @Override
    SkylarkType intersectWith(SkylarkType other) {
      return other;
    }
    @Override public String toString() {
      return "Object";
    }
  }

  /** the Bottom type contains no element */
  private static class Bottom extends Simple {

    private Bottom() {
      super(Empty.class);
    }

    @Override
    SkylarkType intersectWith(SkylarkType other) {
      return this;
    }
    @Override public String toString() {
      return "EmptyType";
    }
  }

  /** a Simple type contains the instance of a Java class */
  @VisibleForSerialization
  static class Simple extends SkylarkType {
    private final Class<?> type;

    private Simple(Class<?> type) {
      this.type = type;
    }

    @Override
    boolean contains(Object value) {
      return value != null && type.isInstance(value);
    }

    @Override
    Class<?> getType() {
      return type;
    }
    @Override public boolean equals(Object other) {
      if (other == null) {
        return false;
      }
      return this == other
          || (this.getClass() == other.getClass() && this.type.equals(((Simple) other).getType()));
    }
    @Override public int hashCode() {
      return 0x513973 + type.hashCode() * 503; // equal underlying types yield the same hashCode
    }
    @Override public String toString() {
      return EvalUtils.getDataTypeNameFromClass(type);
    }

    @Override
    boolean canBeCastTo(Class<?> type) {
      return this.type == type || super.canBeCastTo(type);
    }

    private static final LoadingCache<Class<?>, Simple> simpleCache =
        CacheBuilder.newBuilder()
            .build(
                new CacheLoader<Class<?>, Simple>() {
                  @Override
                  public Simple load(Class<?> type) {
                    return create(type);
                  }
                });

    private static Simple create(Class<?> type) {
      Simple simple;
      if (type == Object.class) {
        // Note that this is a bad encoding for "anything", not for "everything", i.e.
        // for skylark there isn't a type that contains everything, but there's a Top type
        // that corresponds to not knowing yet which more special type it will be.
        simple = TOP;
      } else if (type == Empty.class) {
        simple = BOTTOM;
      } else {
        // Consider all classes that have the same EvalUtils.getSkylarkType() as equivalent,
        // as a substitute to handling inheritance.
        Class<?> skylarkType = EvalUtils.getSkylarkType(type);
        if (skylarkType != type) {
          simple = Simple.forClass(skylarkType);
        } else {
          simple = new Simple(type);
        }
      }
      return simple;
    }

    /**
     * The way to create a Simple type.
     *
     * @param type a Class
     * @return the Simple type that contains exactly the instances of that Class
     */
    // Only call this method from SkylarkType. Calling it from outside SkylarkType leads to
    // circular dependencies in class initialization, showing up as an NPE while initializing NONE.
    // You actually want to call SkylarkType.of().
    private static Simple forClass(Class<?> type) {
      return simpleCache.getUnchecked(type);
    }
  }

  /** Combination of a generic type and an argument type */
  @AutoCodec
  static class Combination extends SkylarkType {
    // For the moment, we can only combine a Simple type with a Simple type,
    // and the first one has to be a Java generic class,
    // and in practice actually one of Sequence or Depset
    private final SkylarkType genericType; // actually always a Simple, for now.
    private final SkylarkType argType; // not always Simple

    @VisibleForSerialization
    Combination(SkylarkType genericType, SkylarkType argType) {
      this.genericType = genericType;
      this.argType = argType;
    }

    @Override
    boolean contains(Object value) {
      // The empty collection is member of compatible types
      if (value == null || !genericType.contains(value)) {
        return false;
      } else {
        SkylarkType valueArgType = getGenericArgType(value);
        return valueArgType == TOP // empty objects are universal
            || argType.includes(valueArgType);
      }
    }

    @Override
    SkylarkType intersectWith(SkylarkType other) {
      // For now, we only accept generics with a single covariant parameter
      if (genericType.equals(other)) {
        return this;
      }
      if (other instanceof Combination) {
        SkylarkType generic = genericType.intersectWith(((Combination) other).getGenericType());
        if (generic == BOTTOM) {
          return BOTTOM;
        }
        SkylarkType arg = intersection(argType, other.getArgType());
        if (arg == BOTTOM) {
          return BOTTOM;
        }
        return Combination.of(generic, arg);
      }
      if (other instanceof Simple) {
        SkylarkType generic = genericType.intersectWith(other);
        if (generic == BOTTOM) {
          return BOTTOM;
        }
        return SkylarkType.of(generic, getArgType());
      }
      return BOTTOM;
    }

    @Override public boolean equals(Object other) {
      if (other == null) {
        return false;
      }
      if (this == other) {
        return true;
      } else if (this.getClass() == other.getClass()) {
        Combination o = (Combination) other;
        return genericType.equals(o.getGenericType())
            && argType.equals(o.getArgType());
      } else {
        return false;
      }
    }
    @Override public int hashCode() {
      // equal underlying types yield the same hashCode
      return 0x20B14A71 + genericType.hashCode() * 1009 + argType.hashCode() * 1013;
    }

    @Override
    Class<?> getType() {
      return genericType.getType();
    }
    SkylarkType getGenericType() {
      return genericType;
    }

    @Override
    SkylarkType getArgType() {
      return argType;
    }
    @Override public String toString() {
      return genericType + " of " + argType + "s";
    }

    private static final Interner<Combination> combinationInterner =
        BlazeInterners.newWeakInterner();

    // TODO(adonovan): eliminate or rename all the 'of' functions.
    // This is an abuse of overloading and "static inheritance".

    static SkylarkType of(SkylarkType generic, SkylarkType argument) {
      // assume all combinations with TOP are the same as the simple type, and canonicalize.
      Preconditions.checkArgument(generic instanceof Simple);
      if (argument == TOP) {
        return generic;
      } else {
        return combinationInterner.intern(new Combination(generic, argument));
      }
    }
  }

  /** Union types, used a lot in "dynamic" languages such as Python or Skylark */
  @AutoCodec
  static class Union extends SkylarkType {
    private final ImmutableList<SkylarkType> types;

    @VisibleForSerialization
    Union(ImmutableList<SkylarkType> types) {
      this.types = types;
    }

    @Override
    boolean contains(Object value) {
      for (SkylarkType type : types) {
        if (type.contains(value)) {
          return true;
        }
      }
      return false;
    }
    @Override public boolean equals(Object other) {
      if (other == null) {
        return false;
      }
      if (this.getClass() == other.getClass()) {
        Union o = (Union) other;
        if (types.containsAll(o.types) && o.types.containsAll(types)) {
          return true;
        }
      }
      return false;
    }
    @Override public int hashCode() {
      // equal underlying types yield the same hashCode
      int h = 0x4104;
      for (SkylarkType type : types) {
        // Important: addition is commutative, like Union
        h += type.hashCode();
      }
      return h;
    }
    @Override public String toString() {
      return Joiner.on(" or ").join(types);
    }

    static List<SkylarkType> addElements(List<SkylarkType> list, SkylarkType type) {
      if (type instanceof Union) {
        list.addAll(((Union) type).types);
      } else if (type != BOTTOM) {
        list.add(type);
      }
      return list;
    }

    @Override
    SkylarkType intersectWith(SkylarkType other) {
      List<SkylarkType> otherTypes = addElements(new ArrayList<>(), other);
      List<SkylarkType> results = new ArrayList<>();
      for (SkylarkType element : types) {
        for (SkylarkType otherElement : otherTypes) {
          addElements(results, intersection(element, otherElement));
        }
      }
      return Union.of(results);
    }

    static SkylarkType of(List<SkylarkType> types) {
      // When making the union of many types,
      // canonicalize them into elementary (non-Union) types,
      // and then eliminate trivially redundant types from the list.

      // list of all types in the input
      ArrayList<SkylarkType> elements = new ArrayList<>();
      for (SkylarkType type : types) {
        addElements(elements, type);
      }

      // canonicalized list of types
      ArrayList<SkylarkType> canonical = new ArrayList<>();

      for (SkylarkType newType : elements) {
        boolean done = false; // done with this element?
        int i = 0;
        for (SkylarkType existingType : canonical) {
          SkylarkType both = intersection(newType, existingType);
          if (newType.equals(both)) { // newType already included
            done = true;
            break;
          } else if (existingType.equals(both)) { // newType supertype of existingType
            canonical.set(i, newType);
            done = true;
            break;
          }
        }
        if (!done) {
          canonical.add(newType);
        }
      }
      if (canonical.isEmpty()) {
        return BOTTOM;
      } else if (canonical.size() == 1) {
        return canonical.get(0);
      } else {
        return new Union(ImmutableList.copyOf(canonical));
      }
    }

    static SkylarkType of(SkylarkType... types) {
      return of(Arrays.asList(types));
    }

    static SkylarkType of(SkylarkType t1, SkylarkType t2) {
      return of(ImmutableList.of(t1, t2));
    }
  }

  // TODO(adonovan): eliminate this function: a value may belong to many types.
  // This function is used to infer the type to use for a Depset from its first
  // element, but this may be unsound in general (e.g. in the presence of sum types).
  static SkylarkType of(Object object) {
    return of(object.getClass());
  }

  /** Returns the type for a Java class. */
  public static SkylarkType of(Class<?> type) {
    if (Depset.class.isAssignableFrom(type)) { // just an optimization
      return SET;
    } else if (BaseFunction.class.isAssignableFrom(type)) {
      return new SkylarkFunctionType("unknown", TOP);
    } else {
      return Simple.forClass(type);
    }
  }

  // TODO(adonovan): these functions abuse overloading and look like sum type constructors.
  // Give them better names such as genericOf(generic, argument)? Or rethink the API.

  private static SkylarkType of(SkylarkType t1, SkylarkType t2) {
    return Combination.of(t1, t2);
  }

  /** Returns the type for a Java parameterized type {@code generic<argument>}. */
  public static SkylarkType of(Class<?> generic, Class<?> argument) {
    return Combination.of(Simple.forClass(generic), Simple.forClass(argument));
  }

  /** A class representing the type of a Skylark function. */
  @AutoCodec
  static final class SkylarkFunctionType extends SkylarkType {
    private final String name;
    @Nullable private final SkylarkType returnType;

    @Override
    SkylarkType intersectWith(SkylarkType other) {
      // This gives the wrong result if both return types are incompatibly updated later!
      if (other instanceof SkylarkFunctionType) {
        SkylarkFunctionType fun = (SkylarkFunctionType) other;
        SkylarkType type1 = returnType == null ? TOP : returnType;
        SkylarkType type2 = fun.returnType == null ? TOP : fun.returnType;
        SkylarkType bothReturnType = intersection(returnType, fun.returnType);
        if (type1.equals(bothReturnType)) {
          return this;
        } else if (type2.equals(bothReturnType)) {
          return fun;
        } else {
          return new SkylarkFunctionType(name, bothReturnType);
        }
      } else {
        return BOTTOM;
      }
    }

    @Override
    Class<?> getType() {
      return BaseFunction.class;
    }
    @Override public String toString() {
      return (returnType == TOP || returnType == null ? "" : returnType + "-returning ")
          + "function";
    }

    @Override
    boolean contains(Object value) {
      // This returns true a bit too much, not looking at the result type.
      return value instanceof BaseFunction;
    }

    static SkylarkFunctionType of(String name, SkylarkType returnType) {
      return new SkylarkFunctionType(name, returnType);
    }

    @VisibleForSerialization
    SkylarkFunctionType(String name, SkylarkType returnType) {
      this.name = name;
      this.returnType = returnType;
    }
  }

  // Utility functions regarding types

  private static SkylarkType getGenericArgType(Object value) {
    if (value instanceof Depset) {
      return ((Depset) value).getContentType();
    } else {
      return TOP;
    }
  }

  /**
   * General purpose type-casting facility.
   *
   * @param value - the actual value of the parameter
   * @param type - the expected Class for the value
   * @param loc - the location info used in the EvalException
   * @param format - a format String
   * @param args - arguments to format, in case there's an exception
   */
  // TODO(adonovan): irrelevant; eliminate.
  public static <T> T cast(Object value, Class<T> type, Location loc, String format, Object... args)
      throws EvalException {
    try {
      return type.cast(value);
    } catch (ClassCastException e) {
      throw new EvalException(loc, String.format(format, args));
    }
  }

  /**
   * General purpose type-casting facility.
   *
   * @param value - the actual value of the parameter
   * @param genericType - a generic class of one argument for the value
   * @param argType - a covariant argument for the generic class
   * @param loc - the location info used in the EvalException
   * @param format - a format String
   * @param args - arguments to format, in case there's an exception
   */
  @SuppressWarnings("unchecked")
  public static <T> T cast(Object value, Class<T> genericType, Class<?> argType,
      Location loc, String format, Object... args) throws EvalException {
    if (of(genericType, argType).contains(value)) {
      return (T) value;
    } else {
      throw new EvalException(loc, String.format(format, args));
    }
  }

  /**
   * Cast a Map object into an Iterable of Map entries of the given key, value types.
   * @param obj the Map object, where null designates an empty map
   * @param keyType the class of map keys
   * @param valueType the class of map values
   * @param what a string indicating what this is about, to include in case of error
   */
  @SuppressWarnings("unchecked")
  public static <KEY_TYPE, VALUE_TYPE> Map<KEY_TYPE, VALUE_TYPE> castMap(Object obj,
      Class<KEY_TYPE> keyType, Class<VALUE_TYPE> valueType, String what)
      throws EvalException {
    if (obj == null) {
      return ImmutableMap.of();
    }
    if (!(obj instanceof Map<?, ?>)) {
      throw new EvalException(
          null,
          String.format(
              "expected a dictionary for '%s' but got '%s' instead",
              what, EvalUtils.getDataTypeName(obj)));
    }

    for (Map.Entry<?, ?> input : ((Map<?, ?>) obj).entrySet()) {
      if (!keyType.isAssignableFrom(input.getKey().getClass())
          || !valueType.isAssignableFrom(input.getValue().getClass())) {
        throw new EvalException(
            null,
            String.format(
                "expected <%s, %s> type for '%s' but got <%s, %s> instead",
                keyType.getSimpleName(),
                valueType.getSimpleName(),
                what,
                EvalUtils.getDataTypeName(input.getKey()),
                EvalUtils.getDataTypeName(input.getValue())));
      }
    }

    return (Map<KEY_TYPE, VALUE_TYPE>) obj;
  }

  // TODO(adonovan): eliminate 4 uses outside this package and make it private.
  // The check is trivial (instanceof) and clients can usually produce a better
  // error in context, without prematurely constructing a description.
  public static void checkType(Object object, Class<?> type, @Nullable Object description)
      throws EvalException {
    if (!type.isInstance(object)) {
      throw new EvalException(
          null,
          Starlark.format(
              "expected type '%r' %sbut got type '%s' instead",
              type,
              description == null ? "" : String.format("for %s ", description),
              EvalUtils.getDataTypeName(object)));
      }
  }
}
