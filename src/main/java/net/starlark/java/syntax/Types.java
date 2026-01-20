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

package net.starlark.java.syntax;

import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Definitions of types.
 *
 * <p><code>
 *   t1, t2 ::= None | bool | int | float | str | object
 *           | t1|t2 | list[t1]
 * </code>
 */
public final class Types {

  // TODO(ilist@): constructed types should probably be interned. In some cases it might help
  // to precompute and memoize StarlarkTypes.getSupertypes.

  /**
   * The Dynamic type of gradual typing; compatible with any other type, but not related by
   * subtyping to any other type.
   */
  public static final StarlarkType ANY = new AnyType();

  /** The top type of the type hierarchy. */
  public static final StarlarkType OBJECT = new ObjectType();

  /** The bottom type of the type hierarchy. */
  public static final StarlarkType NEVER = new NeverType();

  // Primitive types
  public static final StarlarkType NONE = new NoneType();

  public static final StarlarkType BOOL = new BoolType();
  public static final StarlarkType INT = new IntType();
  public static final StarlarkType FLOAT = new FloatType();
  public static final StarlarkType STR = new StrType();

  // A frequently used function without parameters, that returns Any.
  public static final CallableType NO_PARAMS_CALLABLE =
      callable(ImmutableList.of(), ImmutableList.of(), 0, 0, ImmutableSet.of(), null, null, ANY);

  private Types() {} // uninstantiable

  public static final ImmutableMap<String, TypeConstructor> TYPE_UNIVERSE = makeTypeUniverse();

  private static ImmutableMap<String, TypeConstructor> makeTypeUniverse() {
    ImmutableMap.Builder<String, TypeConstructor> env = ImmutableMap.builder();
    env //
        .put("Any", wrapType("Any", ANY))
        .put("object", wrapType("object", OBJECT))
        .put("None", wrapType("None", NONE))
        .put("bool", wrapType("bool", BOOL))
        .put("int", wrapType("int", INT))
        .put("float", wrapType("float", FLOAT))
        .put("str", wrapType("str", STR))
        .put("list", wrapTypeConstructor("list", Types::list))
        .put("dict", wrapTypeConstructor("dict", Types::dict))
        .put("set", wrapTypeConstructor("set", Types::set))
        .put("tuple", wrapTupleConstructor())
        .put("Collection", wrapTypeConstructor("Collection", Types::collection))
        .put("Sequence", wrapTypeConstructor("Sequence", Types::sequence))
        .put("Mapping", wrapTypeConstructor("Mapping", Types::mapping));
    return env.buildOrThrow();
  }

  // hashCode and equals implementation is a workaround for serialization code that may duplicate
  // otherwise singletons
  private static final class AnyType extends StarlarkType {
    @Override
    public String toString() {
      return "Any";
    }

    @Override
    public int hashCode() {
      return AnyType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof AnyType;
    }
  }

  private static final class ObjectType extends StarlarkType {
    @Override
    public String toString() {
      return "object";
    }

    @Override
    public int hashCode() {
      return ObjectType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof ObjectType;
    }
  }

  private static final class NeverType extends StarlarkType {
    @Override
    public String toString() {
      return "Never";
    }

    @Override
    public int hashCode() {
      return NeverType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof NeverType;
    }
  }

  private static final class NoneType extends StarlarkType {
    @Override
    public String toString() {
      return "None";
    }

    @Override
    public int hashCode() {
      return NoneType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof NoneType;
    }
  }

  private static final class BoolType extends StarlarkType {
    @Override
    public String toString() {
      return "bool";
    }

    @Override
    public int hashCode() {
      return BoolType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof BoolType;
    }
  }

  private static final class IntType extends StarlarkType {
    @Override
    public String toString() {
      return "int";
    }

    @Override
    public int hashCode() {
      return IntType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof IntType;
    }
  }

  private static final class FloatType extends StarlarkType { // Float clashes with java.lang.Float
    @Override
    public String toString() {
      return "float";
    }

    @Override
    public int hashCode() {
      return FloatType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof FloatType;
    }
  }

  private static final class StrType extends StarlarkType {
    @Override
    public String toString() {
      return "str";
    }

    @Override
    public int hashCode() {
      return StrType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof StrType;
    }
  }

  /** Construct a CallableType representing a Starlark Function */
  public static CallableType callable(
      ImmutableList<String> parameterNames,
      ImmutableList<StarlarkType> parameterTypes,
      int numPositionalOnlyParameters,
      int numPositionalParameters,
      ImmutableSet<String> mandatoryParams,
      @Nullable StarlarkType varargsType,
      @Nullable StarlarkType kwargsType,
      StarlarkType returns) {
    Preconditions.checkArgument(
        parameterNames.size() == parameterTypes.size(),
        "%s != %s",
        parameterNames.size(),
        parameterTypes.size());
    return new AutoValue_Types_GeneralCallableType(
        parameterNames,
        parameterTypes,
        numPositionalOnlyParameters,
        numPositionalParameters,
        mandatoryParams,
        varargsType,
        kwargsType,
        returns);
  }

  /**
   * An interface for the general Starlark callable.
   *
   * <p>There are 3 flavours of parameters:
   *
   * <ul>
   *   <li>positional-only (can't be passed with a keyword),
   *   <li>ordinary (can be passed by position or with a keyword) and
   *   <li>keyword-only parameters.
   * </ul>
   *
   * The interface describes them as follows:
   *
   * <ul>
   *   <li>Their types are stored consecutively in <code>parameterTypes</code>.
   *   <li>The list <code>parameterNames</code> matches <code>parameterTypes</code>. (Even
   *       positional-only parameters have names.)
   *   <li><code>numPositionalOnlyParameters</code> counts positional-only arguments.
   *   <li><code>numPositionalParameters</code> counts both positional-only and ordinary arguments.
   * </ul>
   *
   * <p>Special parameters {@code *args} and {@code **kwargs} are stored separately. If they are
   * absent, they are set to {@code null}.
   *
   * <p>Mandatory parameters (non-special parameters without default values) are stored as an
   * ordered set.
   *
   * <p>The return type is marked as Any if not annotated.
   */
  public abstract static class CallableType extends StarlarkType {

    public abstract ImmutableList<String> getParameterNames();

    public abstract ImmutableList<StarlarkType> getParameterTypes();

    public abstract int getNumPositionalOnlyParameters();

    public abstract int getNumPositionalParameters();

    public abstract ImmutableSet<String> getMandatoryParameters();

    @Nullable
    public abstract StarlarkType getVarargsType();

    @Nullable
    public abstract StarlarkType getKwargsType();

    public abstract StarlarkType getReturnType();

    public StarlarkType getParameterTypeByPos(int i) {
      return getParameterTypes().get(i);
    }

    @Override
    public String toString() {
      // Approximate representation of the type - as much as Callable can do
      return "Callable[["
          + getParameterTypes().stream().map(StarlarkType::toString).collect(joining(", "))
          + "], "
          + getReturnType()
          + "]";
    }

    /** Returns a complete string representation of the type */
    public String toSignatureString() {
      ImmutableList.Builder<String> params = ImmutableList.builder();

      // positional parameters
      int i = 0;
      for (; i < getNumPositionalOnlyParameters(); i++) {
        String name = getParameterNames().get(i);
        StarlarkType type = getParameterTypeByPos(i);
        if (getMandatoryParameters().contains(name)) {
          params.add(type.toString());
        } else {
          params.add("[" + type + "]");
        }
      }

      if (i > 0) { // if there were positional-only parameters, we need to separate them
        params.add("/");
      }

      for (; i < getNumPositionalParameters(); i++) {
        String name = getParameterNames().get(i);
        StarlarkType type = getParameterTypeByPos(i);
        if (getMandatoryParameters().contains(name)) {
          params.add(name + ": " + type);
        } else {
          params.add(name + ": [" + type + "]");
        }
      }

      if (getVarargsType() != null) {
        params.add("*args: " + getVarargsType());
      } else if (i < getParameterTypes().size()) { // if there are going to be kwonly params
        params.add("*");
      }

      // keyword parameters
      for (; i < getParameterTypes().size(); i++) {
        String name = getParameterNames().get(i);
        String type = getParameterTypeByPos(i).toString();
        if (getMandatoryParameters().contains(name)) {
          params.add(name + ": " + type);
        } else {
          params.add(name + ": [" + type + "]");
        }
      }

      if (getKwargsType() != null) {
        params.add("**kwargs: " + getKwargsType());
      }

      ImmutableList<String> paramList = params.build();
      return "(" + String.join(", ", paramList) + ") -> " + getReturnType();
    }
  }

  // About 0.1% memory regression may be removed by specializing GeneralCallableType for function
  // without positional-only parameter and by retrieving parameter names from StarlarkFunction
  @AutoValue
  abstract static class GeneralCallableType extends CallableType {}

  /**
   * Constructs a union type.
   *
   * <p>If the types sets contains another Union type it's flattened. Duplicates are removed.
   *
   * <p>If types set contains Object type it's simplified to Object type. If the set contains a
   * single element, it is returned instead of constructing a union.
   */
  public static StarlarkType union(StarlarkType... types) {
    return union(ImmutableSet.copyOf(types));
  }

  /** Constructs a union type. */
  // TODO: #28043 - Seems more appropriate to use List<StarlarkType> for the param and let this
  // factory method take care of deduplication. For the moment we have a convenience overload below.
  public static StarlarkType union(ImmutableSet<StarlarkType> types) {
    ImmutableSet.Builder<StarlarkType> subtypesBuilder = ImmutableSet.builder();
    // Unions are flattened
    for (StarlarkType type : types) {
      if (type instanceof UnionType union) {
        subtypesBuilder.addAll(union.getTypes());
      } else {
        subtypesBuilder.add(type);
      }
    }
    ImmutableSet<StarlarkType> subtypes = subtypesBuilder.build();
    if (subtypes.contains(Types.OBJECT)) {
      return Types.OBJECT;
    }
    if (subtypes.size() == 1) {
      return subtypes.iterator().next();
    } else if (subtypes.isEmpty()) {
      return Types.NEVER;
    }
    return new AutoValue_Types_UnionType(subtypes);
  }

  public static StarlarkType union(List<StarlarkType> types) {
    return union(ImmutableSet.copyOf(types));
  }

  /**
   * Union type
   *
   * <p>Unions with zero or one type are disallowed. See {@link Types#union}.
   */
  @AutoValue
  public abstract static class UnionType extends StarlarkType {
    public abstract ImmutableSet<StarlarkType> getTypes();

    @Override
    public final String toString() {
      return getTypes().stream().map(StarlarkType::toString).collect(joining("|"));
    }
  }

  public static ListType list(StarlarkType elementType) {
    return new AutoValue_Types_ListType(elementType);
  }

  /** List type */
  @AutoValue
  public abstract static class ListType extends StarlarkType {
    public abstract StarlarkType getElementType();

    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(sequence(getElementType()), collection(getElementType()));
    }

    @Override
    public final String toString() {
      return "list[" + getElementType() + "]";
    }
  }

  public static DictType dict(StarlarkType keyType, StarlarkType valueType) {
    return new AutoValue_Types_DictType(keyType, valueType);
  }

  /** Dict type */
  @AutoValue
  public abstract static class DictType extends StarlarkType {
    public abstract StarlarkType getKeyType();

    public abstract StarlarkType getValueType();

    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(collection(getKeyType()), mapping(getKeyType(), getValueType()));
    }

    @Override
    public final String toString() {
      return "dict[" + getKeyType() + ", " + getValueType() + "]";
    }
  }

  public static SetType set(StarlarkType elementType) {
    return new AutoValue_Types_SetType(elementType);
  }

  /** Set type */
  @AutoValue
  public abstract static class SetType extends StarlarkType {
    public abstract StarlarkType getElementType();

    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(collection(getElementType()));
    }

    @Override
    public final String toString() {
      return "set[" + getElementType() + "]";
    }
  }

  public static TupleType tuple(ImmutableList<StarlarkType> elementTypes) {
    return new AutoValue_Types_TupleType(elementTypes);
  }

  /** Tuple type of a fixed length. */
  @AutoValue
  public abstract static class TupleType extends StarlarkType {
    public abstract ImmutableList<StarlarkType> getElementTypes();

    @Override
    public List<StarlarkType> getSupertypes() {
      StarlarkType elementType = union(ImmutableSet.copyOf(getElementTypes()));
      return ImmutableList.of(sequence(elementType), collection(elementType));
    }

    @Override
    public final String toString() {
      return "tuple["
          + getElementTypes().stream().map(StarlarkType::toString).collect(joining(", "))
          + "]";
    }
  }

  /** Collection type */
  public static CollectionType collection(StarlarkType elementType) {
    return new AutoValue_Types_CollectionType(elementType);
  }

  /** Collection type */
  @AutoValue
  public abstract static class CollectionType extends StarlarkType {
    public abstract StarlarkType getElementType();

    @Override
    public final String toString() {
      return "Collection[" + getElementType() + "]";
    }
  }

  /** Sequence type */
  public static SequenceType sequence(StarlarkType elementType) {
    return new AutoValue_Types_SequenceType(elementType);
  }

  /** Sequence type */
  @AutoValue
  public abstract static class SequenceType extends StarlarkType {
    public abstract StarlarkType getElementType();

    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(collection(getElementType()));
    }

    @Override
    public final String toString() {
      return "Sequence[" + getElementType() + "]";
    }
  }

  /** Mapping type */
  public static MappingType mapping(StarlarkType keyType, StarlarkType valueType) {
    return new AutoValue_Types_MappingType(keyType, valueType);
  }

  /** Mapping type */
  @AutoValue
  public abstract static class MappingType extends StarlarkType {
    public abstract StarlarkType getKeyType();

    public abstract StarlarkType getValueType();

    @Override
    public final String toString() {
      return "Mapping[" + getKeyType() + ", " + getValueType() + "]";
    }
  }

  static TypeConstructor wrapType(String name, StarlarkType type) {
    return argsTuple -> {
      if (!argsTuple.isEmpty()) {
        throw new TypeConstructor.Failure(String.format("'%s' does not accept arguments", name));
      }
      return type;
    };
  }

  static TypeConstructor wrapTypeConstructor(
      String name, Function<StarlarkType, StarlarkType> constructor) {
    return argsTuple -> {
      if (argsTuple.size() != 1) {
        throw new TypeConstructor.Failure(
            String.format("%s[] accepts exactly 1 argument but got %d", name, argsTuple.size()));
      }
      if (!(argsTuple.get(0) instanceof StarlarkType type)) {
        throw new TypeConstructor.Failure(
            String.format(
                "in application to %s, got '%s', expected a type", name, argsTuple.get(0)));
      }
      return constructor.apply(type);
    };
  }

  static TypeConstructor wrapTypeConstructor(
      String name, BiFunction<StarlarkType, StarlarkType, StarlarkType> constructor) {
    return argsTuple -> {
      if (argsTuple.size() != 2) {
        throw new TypeConstructor.Failure(
            String.format("%s[] accepts exactly 2 arguments but got %d", name, argsTuple.size()));
      }
      if (!(argsTuple.get(0) instanceof StarlarkType keyType)) {
        throw new TypeConstructor.Failure(
            String.format(
                "in application to %s, got '%s', expected a type", name, argsTuple.get(0)));
      }
      if (!(argsTuple.get(1) instanceof StarlarkType valueType)) {
        throw new TypeConstructor.Failure(
            String.format(
                "in application to %s, got '%s', expected a type", name, argsTuple.get(1)));
      }
      return constructor.apply(keyType, valueType);
    };
  }

  private static final TypeConstructor wrapTupleConstructor() {
    // This is a function instead of a constant, so that the order of evaluation doesn't depend on
    // the position in the class.
    return argsTuple -> {
      ImmutableList.Builder<StarlarkType> elementTypes =
          ImmutableList.builderWithExpectedSize(argsTuple.size());
      for (Object arg : argsTuple) {
        if (!(arg instanceof StarlarkType type)) {
          throw new TypeConstructor.Failure(
              String.format("in application to tuple, got '%s', expected a type", arg));
        }
        elementTypes.add(type);
      }
      return tuple(elementTypes.build());
    };
  }
}
