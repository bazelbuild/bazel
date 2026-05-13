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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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

  // A frequently-used union `int | float`.
  public static final UnionType NUMERIC = (UnionType) union(INT, FLOAT);
  // A frequently-used empty tuple type.
  public static final FixedLengthTupleType EMPTY_TUPLE = tuple(ImmutableList.of());
  // A frequently-used arbitrary collection.
  public static final CollectionType COLLECTION_OF_ANY = collection(ANY);

  // A frequently used function without parameters, that returns Any.
  public static final CallableType NO_PARAMS_CALLABLE =
      callable(ImmutableList.of(), ImmutableList.of(), 0, 0, ImmutableSet.of(), null, null, ANY);

  public static final TypeConstructor ANY_CONSTRUCTOR = wrapType("Any", ANY);
  public static final TypeConstructor OBJECT_CONSTRUCTOR = wrapType("object", OBJECT);
  public static final TypeConstructor NONE_CONSTRUCTOR = wrapType("None", NONE);
  public static final TypeConstructor BOOL_CONSTRUCTOR = wrapType("bool", BOOL);
  public static final TypeConstructor INT_CONSTRUCTOR = wrapType("int", INT);
  public static final TypeConstructor FLOAT_CONSTRUCTOR = wrapType("float", FLOAT);
  public static final TypeConstructor STR_CONSTRUCTOR = wrapType("str", STR);
  public static final TypeConstructor LIST_CONSTRUCTOR = wrapTypeConstructor("list", Types::list);
  public static final TypeConstructor DICT_CONSTRUCTOR = wrapTypeConstructor("dict", Types::dict);
  public static final TypeConstructor SET_CONSTRUCTOR = wrapTypeConstructor("set", Types::set);
  public static final TypeConstructor TUPLE_CONSTRUCTOR = wrapTupleConstructor();
  public static final TypeConstructor COLLECTION_CONSTRUCTOR =
      wrapTypeConstructor("Collection", Types::collection);
  public static final TypeConstructor SEQUENCE_CONSTRUCTOR =
      wrapTypeConstructor("Sequence", Types::sequence);
  public static final TypeConstructor MAPPING_CONSTRUCTOR =
      wrapTypeConstructor("Mapping", Types::mapping);
  public static final TypeConstructor STRUCT_CONSTRUCTOR = wrapStructConstructor();

  private Types() {} // uninstantiable

  public static final ImmutableMap<String, TypeConstructor> TYPE_UNIVERSE = makeTypeUniverse();

  // Note that STRUCT_CONSTRUCTOR is not in the type universe; applications are responsible for
  // adding it if needed.
  private static ImmutableMap<String, TypeConstructor> makeTypeUniverse() {
    ImmutableMap.Builder<String, TypeConstructor> env = ImmutableMap.builder();
    env //
        .put("Any", ANY_CONSTRUCTOR)
        .put("object", OBJECT_CONSTRUCTOR)
        .put("None", NONE_CONSTRUCTOR)
        .put("bool", BOOL_CONSTRUCTOR)
        .put("int", INT_CONSTRUCTOR)
        .put("float", FLOAT_CONSTRUCTOR)
        .put("str", STR_CONSTRUCTOR)
        .put("list", LIST_CONSTRUCTOR)
        .put("dict", DICT_CONSTRUCTOR)
        .put("set", SET_CONSTRUCTOR)
        .put("tuple", TUPLE_CONSTRUCTOR)
        .put("Collection", COLLECTION_CONSTRUCTOR)
        .put("Sequence", SEQUENCE_CONSTRUCTOR)
        .put("Mapping", MAPPING_CONSTRUCTOR);
    return env.buildOrThrow();
  }

  // hashCode and equals implementation is a workaround for serialization code that may duplicate
  // otherwise singletons
  private static final class AnyType extends StarlarkType {
    // Singleton.
    private AnyType() {}

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

    @Override
    public StarlarkType getField(String name, TypeContext context) {
      return ANY;
    }

    // TODO: #27370 - we may want to infer a more precise type when one of the operands is non-Any.
    // (For example, we could infer that int % Any is int | float; on the other hand, Any % int
    // could also be a string, since % is also a string substitution operator.) Requires a registry
    // of which types (including those of application-defined net.starlark.java.eval.HasBinary
    // values) support which binary operators. This would also imply that the inferred type of
    // `Any <op> T` could be application-dependent even if T is a universal built-in type.
    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        case IN, NOT_IN ->
            // If we are the LHS, fall through to RHS's inferBinaryOperator; RHS determines whether
            // it is membership-testable.
            // If we are the RHS, act as a membership-testable type that allows any LHS (e.g. list)
            // and return bool.
            thisLeft ? null : Types.BOOL;
        default -> ANY;
      };
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      // Instead of enumerating all comparable types here, allow StarlarkType#comparable to defer to
      // that.isComparable(ANY).
      return that.equals(ANY);
    }

    @Override
    public boolean hasSetIndex() {
      return true;
    }

    @Override
    public boolean hasSetField() {
      return true;
    }
  }

  private static final class ObjectType extends StarlarkType {
    // Singleton.
    private ObjectType() {}

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
    // Singleton.
    private NeverType() {}

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

    @Override
    protected boolean isComparable(StarlarkType that) {
      // Regard Never - as the bottom type - to be comparable to anything; in particular, this
      // allows empty lists (i.e. list[Never]) to be comparable to arbitrary non-empty lists.
      return true;
    }

    @Override
    public boolean hasSetIndex() {
      return true;
    }

    @Override
    public boolean hasSetField() {
      return true;
    }
  }

  private static final class NoneType extends StarlarkType {
    // Singleton.
    private NoneType() {}

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
    // Singleton.
    private BoolType() {}

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

    @Override
    protected boolean isComparable(StarlarkType that) {
      return StarlarkType.assignableFrom(Types.BOOL, that);
    }
  }

  private static final class IntType extends StarlarkType {
    // Singleton.
    private IntType() {}

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

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        case PLUS, MINUS, PERCENT, SLASH_SLASH -> NUMERIC.getTypes().contains(that) ? that : null;
        case SLASH -> NUMERIC.getTypes().contains(that) ? Types.FLOAT : null;
        case STAR ->
            // Repetition operator (int * str, int * list, etc.) is assumed to be symmetric and
            // implemented by the rhs, so defer to rhs for non-numeric case.
            NUMERIC.getTypes().contains(that) ? that : null;
        case AMPERSAND, CARET, GREATER_GREATER, LESS_LESS, PIPE ->
            that.equals(Types.INT) ? Types.INT : null;
        default -> null;
      };
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      return StarlarkType.assignableFrom(NUMERIC, that);
    }
  }

  private static final class FloatType extends StarlarkType {
    // Singleton.
    private FloatType() {}

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

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        case PLUS, MINUS, PERCENT, SLASH, SLASH_SLASH, STAR ->
            NUMERIC.getTypes().contains(that) ? Types.FLOAT : null;
        default -> null;
      };
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      return StarlarkType.assignableFrom(NUMERIC, that);
    }
  }

  private static final class StrType extends StarlarkType {
    // Singleton.
    private StrType() {}

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

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        case PLUS -> that.equals(STR) ? STR : null;
        case PERCENT ->
            // String substitution allows anything on the RHS
            thisLeft ? STR : null;
        case STAR -> that.equals(INT) ? STR : null;
        case IN, NOT_IN ->
            // If we are LHS, defer to the RHS.
            // If we are RHS, explicitly handle Any since AnyType.inferBinaryOperator defers to us.
            !thisLeft && (that.equals(STR) || that.equals(ANY)) ? BOOL : null;
        default -> null;
      };
    }

    @Override
    @Nullable
    public StarlarkType getField(String name, TypeContext context) {
      return context.getStrFieldType(name);
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      return that.equals(STR) || that.equals(ANY);
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
   * <p>If the types set contains another Union type it's flattened. Duplicates are removed.
   * Occurrences of Never are removed.
   *
   * <p>If types set contains Object type it's simplified to Object type. If the set contains a
   * single element, it is returned instead of constructing a union. And if the set is empty, Never
   * is returned.
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
      } else if (!type.equals(Types.NEVER)) {
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
    if (types.size() == 1) {
      // Optimize the common case.
      return types.getFirst();
    }
    return union(ImmutableSet.copyOf(types));
  }

  /** Returns the list of a union's types, or a singleton list if {@code type} is not a union. */
  public static ImmutableCollection<StarlarkType> unfoldUnion(StarlarkType type) {
    if (type instanceof Types.UnionType unionType) {
      return unionType.getTypes();
    }
    return ImmutableList.of(type);
  }

  /**
   * Union type
   *
   * <p>Unions must contain at least two types, none of which may be Never or Object. See {@link
   * Types#union}.
   */
  @AutoValue
  public abstract static class UnionType extends StarlarkType {
    public abstract ImmutableSet<StarlarkType> getTypes();

    @Override
    public final String toString() {
      return getTypes().stream().map(StarlarkType::toString).collect(joining("|"));
    }

    @Override
    public StarlarkType toLvalue() {
      return union(getTypes().stream().map(StarlarkType::toLvalue).collect(toImmutableSet()));
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      return getTypes().stream().allMatch(type -> StarlarkType.comparable(type, that));
    }

    @Override
    @Nullable
    public StarlarkType getField(String name, TypeContext context) {
      ArrayList<StarlarkType> resultTypes = new ArrayList<>(getTypes().size());
      for (StarlarkType type : getTypes()) {
        StarlarkType result = type.getField(name, context);
        if (result == null) {
          return null;
        }
        resultTypes.add(result);
      }
      return union(resultTypes);
    }

    @Override
    public boolean hasSetIndex() {
      return getTypes().stream().allMatch(StarlarkType::hasSetIndex);
    }

    @Override
    public boolean hasSetField() {
      return getTypes().stream().allMatch(StarlarkType::hasSetField);
    }
  }

  public static ListType list(StarlarkType elementType) {
    return new AutoValue_Types_ListType(elementType);
  }

  /**
   * Constructs a list rvalue type. Only for literals and anonymous temporary values.
   *
   * <p>Like all rvalue types, this type MUST NOT be used as or in the type of any variable or
   * parameter; and it MUST NOT be inferred as or in a type parameter of a generic function.
   */
  public static ListRvalueType listRvalue(StarlarkType elementType) {
    return new AutoValue_Types_ListRvalueType(elementType);
  }

  /** List type */
  public abstract static sealed class BaseListType extends AbstractSequenceType
      permits ListType, ListRvalueType {
    @Override
    public final String toString() {
      return "list[" + getElementType() + "]";
    }

    @Override
    public ListType toLvalue() {
      return list(getElementType().toLvalue());
    }

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        case PLUS ->
            that instanceof BaseListType thatList
                ? listRvalue(union(getElementType(), thatList.getElementType()))
                : null;
        case STAR -> that.equals(Types.INT) ? this.toRvalue() : null;
        default -> super.inferBinaryOperator(operator, that, thisLeft);
      };
    }

    @Override
    @Nullable
    public StarlarkType getField(String name, TypeContext context) {
      return context.getListFieldType(name);
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      if (that.equals(Types.ANY)) {
        return true;
      } else if (that instanceof BaseListType thatList) {
        return comparable(getElementType(), thatList.getElementType());
      }
      return false;
    }

    @Override
    public boolean hasSetIndex() {
      return true;
    }
  }

  /**
   * The type of a new, unaliased list value; for example, a list literal or the result of a binary
   * operator which has not yet been assigned.
   */
  @AutoValue
  public abstract static non-sealed class ListRvalueType extends BaseListType {
    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(
          list(getElementType()), sequence(getElementType()), collection(getElementType()));
    }

    @Override
    public ListRvalueType toRvalue() {
      return this;
    }

    @Override
    protected boolean isRvalueAssignableTo(AbstractCollectionType that) {
      // Covariant in element type. Assignable only to types having a constructor which is a
      // constructor of one of this type's supertypes (in particular: not assignable to dicts,
      // sets, or application-defined types).
      // TODO: #27370 - when we have type deconstruction, replace `instanceof` checks below with
      // deconstruction of getSupertypes().
      return (that instanceof BaseListType
              || that instanceof SequenceType
              || that instanceof CollectionType)
          && StarlarkType.assignableFrom(that.getElementType(), this.getElementType());
    }
  }

  /**
   * The type of a potentially aliased list value; for example, the value of a variable, or nested
   * in a variable's compound value.
   */
  @AutoValue
  public abstract static non-sealed class ListType extends BaseListType {
    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(sequence(getElementType()), collection(getElementType()));
    }

    @Override
    public ListRvalueType toRvalue() {
      return listRvalue(getElementType());
    }
  }

  public static DictType dict(StarlarkType keyType, StarlarkType valueType) {
    return new AutoValue_Types_DictType(keyType, valueType);
  }

  /**
   * Constructs a dict rvalue type. Only for literals and anonymous temporary values.
   *
   * <p>Like all rvalue types, this type MUST NOT be used as or in the type of any variable or
   * parameter; and it MUST NOT be inferred as or in a type parameter of a generic function.
   */
  public static DictRvalueType dictRvalue(StarlarkType keyType, StarlarkType valueType) {
    return new AutoValue_Types_DictRvalueType(keyType, valueType);
  }

  /** Dict type */
  public abstract static sealed class BaseDictType extends AbstractMappingType
      permits DictType, DictRvalueType {
    @Override
    public abstract StarlarkType getKeyType();

    @Override
    public abstract StarlarkType getValueType();

    @Override
    public final String toString() {
      return "dict[" + getKeyType() + ", " + getValueType() + "]";
    }

    @Override
    public DictType toLvalue() {
      return dict(getKeyType().toLvalue(), getValueType().toLvalue());
    }

    @Override
    @Nullable
    public StarlarkType getField(String name, TypeContext context) {
      return context.getDictFieldType(name);
    }

    @Override
    public boolean hasSetIndex() {
      return true;
    }
  }

  /**
   * The type of a new, unaliased dict value; for example, a dict literal or the result of a binary
   * operator which has not yet been assigned.
   */
  @AutoValue
  public abstract static non-sealed class DictRvalueType extends BaseDictType {
    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(
          dict(getKeyType(), getValueType()),
          mapping(getKeyType(), getValueType()),
          collection(getKeyType()));
    }

    @Override
    public DictRvalueType toRvalue() {
      return this;
    }

    @Override
    protected boolean isMappingRvalueAssignableTo(AbstractMappingType that) {
      // Covariant in both key and value types. This differs from Mapping, which is covariant only
      // in the value type, because we need to be able to assign e.g. an empty dict having Never key
      // type. Mapping avoids covariance in keys in order to catch type errors at lookups, but
      // that's not a concern for rvalue dicts since this method is only checked upon promotion to
      // lvalues, not at indexing expressions.
      // Assignable only to types having a constructor which is a constructor of one of this type's
      // supertypes (in particular: not assignable to sequences, sets, or application-defined
      // types).
      // TODO: #27370 - when we have type deconstruction, replace `instanceof` checks below with
      // deconstruction of getSupertypes().
      return (that instanceof BaseDictType || that instanceof MappingType)
          && StarlarkType.assignableFrom(that.getKeyType(), getKeyType())
          && StarlarkType.assignableFrom(that.getValueType(), getValueType());
    }
  }

  /**
   * The type of a potentially aliased dict value; for example, the value of a variable, or nested
   * in a variable's compound value.
   */
  @AutoValue
  public abstract static non-sealed class DictType extends BaseDictType {
    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(mapping(getKeyType(), getValueType()), collection(getKeyType()));
    }

    @Override
    public DictRvalueType toRvalue() {
      return dictRvalue(getKeyType(), getValueType());
    }
  }

  public static SetType set(StarlarkType elementType) {
    return new AutoValue_Types_SetType(elementType);
  }

  /** Set type */
  // TODO: #27370 - add Rvalue version (same as for ListType and DictType) and have the {@code
  // set()} built-in function return an rvalue. To be useful, this would first require generics
  // support for StarlarkMethod.
  @AutoValue
  public abstract static class SetType extends AbstractCollectionType {
    @Override
    public abstract StarlarkType getElementType();

    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(collection(getElementType()));
    }

    @Override
    public final String toString() {
      return "set[" + getElementType() + "]";
    }

    @Override
    @Nullable
    public StarlarkType getField(String name, TypeContext context) {
      return context.getSetFieldType(name);
    }

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        case AMPERSAND, MINUS ->
            // TODO: #27370 - we may want to tighten the type of a set intersection, but it's
            // non-trivial.
            that instanceof SetType ? this : null;
        case CARET, PIPE ->
            that instanceof SetType thatSet
                ? set(union(getElementType(), thatSet.getElementType()))
                : null;
        default -> super.inferBinaryOperator(operator, that, thisLeft);
      };
    }

    @Override
    public SetType toLvalue() {
      return set(getElementType().toLvalue());
    }
  }

  public static FixedLengthTupleType tuple(ImmutableList<StarlarkType> elementTypes) {
    return new AutoValue_Types_FixedLengthTupleType(elementTypes);
  }

  public static FixedLengthTupleType tuple(StarlarkType first, StarlarkType... rest) {
    return tuple(ImmutableList.<StarlarkType>builder().add(first).add(rest).build());
  }

  public static HomogeneousTupleType homogeneousTuple(StarlarkType elementType) {
    return new AutoValue_Types_HomogeneousTupleType(elementType);
  }

  /** Tuple type. */
  public abstract static sealed class TupleType extends AbstractSequenceType
      permits FixedLengthTupleType, HomogeneousTupleType {
    /** Returns the type of this tuple concatenated with another. */
    abstract TupleType concatenate(TupleType rhs);

    /** Returns the type of this tuple repeated. */
    abstract TupleType repeat(int times);

    /** Returns the homogeneous version of this tuple type. */
    public abstract HomogeneousTupleType toHomogeneous();

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        case PLUS -> that instanceof TupleType rhsTuple ? concatenate(rhsTuple) : null;
        // Special case handled by TypeChecker.inferTupleRepetition.
        case STAR -> null;
        default -> super.inferBinaryOperator(operator, that, thisLeft);
      };
    }
  }

  /** Tuple type of a fixed length. */
  @AutoValue
  public abstract static non-sealed class FixedLengthTupleType extends TupleType {
    public abstract ImmutableList<StarlarkType> getElementTypes();

    @Override
    public StarlarkType getElementType() {
      return union(getElementTypes());
    }

    @Override
    public boolean assignableFromHook(StarlarkType t) {
      if (!(t instanceof FixedLengthTupleType that)) {
        return false;
      }
      // Covariant in each element type; the number of elements must match exactly.
      if (this.getElementTypes().size() != that.getElementTypes().size()) {
        return false;
      }
      for (int i = 0; i < this.getElementTypes().size(); i++) {
        if (!StarlarkType.assignableFrom(
            this.getElementTypes().get(i), that.getElementTypes().get(i))) {
          return false;
        }
      }
      return true;
    }

    @Override
    public List<StarlarkType> getSupertypes() {
      HomogeneousTupleType homogeneous = toHomogeneous();
      return ImmutableList.of(
          homogeneous,
          sequence(homogeneous.getElementType()),
          collection(homogeneous.getElementType()));
    }

    @Override
    public final String toString() {
      return String.format(
          "tuple[%s]",
          getElementTypes().isEmpty()
              ? "()"
              : getElementTypes().stream().map(StarlarkType::toString).collect(joining(", ")));
    }

    @Override
    TupleType concatenate(TupleType rhs) {
      if (rhs instanceof FixedLengthTupleType rhsFixedLength) {
        return tuple(
            ImmutableList.<StarlarkType>builder()
                .addAll(getElementTypes())
                .addAll(rhsFixedLength.getElementTypes())
                .build());
      } else {
        return toHomogeneous().concatenate(rhs);
      }
    }

    @Override
    FixedLengthTupleType repeat(int times) {
      ImmutableList.Builder<StarlarkType> builder = ImmutableList.builder();
      for (int i = 0; i < times; i++) {
        builder.addAll(getElementTypes());
      }
      return tuple(builder.build());
    }

    @Override
    public HomogeneousTupleType toHomogeneous() {
      return homogeneousTuple(union(getElementTypes()));
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      if (that.equals(Types.ANY)) {
        return true;
      } else if (that instanceof FixedLengthTupleType thatTuple) {
        int commonLength = Math.min(getElementTypes().size(), thatTuple.getElementTypes().size());
        for (int i = 0; i < commonLength; i++) {
          if (!comparable(getElementTypes().get(i), thatTuple.getElementTypes().get(i))) {
            return false;
          }
        }
        return true;
      }
      // Comparison with HomogeneousTupleType defers to HomogeneousTupleType.
      return false;
    }

    @Override
    public FixedLengthTupleType toLvalue() {
      return tuple(
          getElementTypes().stream().map(StarlarkType::toLvalue).collect(toImmutableList()));
    }
  }

  /** Tuple type of an indeterminate length. */
  @AutoValue
  public abstract static non-sealed class HomogeneousTupleType extends TupleType {
    @Override
    public abstract StarlarkType getElementType();

    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(sequence(getElementType()), collection(getElementType()));
    }

    @Override
    public boolean assignableFromHook(StarlarkType t) {
      if (!(t instanceof HomogeneousTupleType that)) {
        return false;
      }
      // Covariant in element type.
      return StarlarkType.assignableFrom(this.getElementType(), that.getElementType());
    }

    @Override
    public final String toString() {
      return "tuple[" + getElementType() + ", ...]";
    }

    @Override
    HomogeneousTupleType concatenate(TupleType rhs) {
      return rhs instanceof HomogeneousTupleType rhsHomogeneous
          ? homogeneousTuple(union(getElementType(), rhsHomogeneous.getElementType()))
          : concatenate(rhs.toHomogeneous());
    }

    @Override
    TupleType repeat(int times) {
      return times > 0 ? this : Types.EMPTY_TUPLE;
    }

    @Override
    public HomogeneousTupleType toHomogeneous() {
      return this;
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      if (that.equals(Types.ANY)) {
        return true;
      } else if (that instanceof TupleType thatTuple) {
        return comparable(getElementType(), thatTuple.toHomogeneous().getElementType());
      }
      return false;
    }

    @Override
    public HomogeneousTupleType toLvalue() {
      return homogeneousTuple(getElementType().toLvalue());
    }
  }

  /** Collection type */
  public static CollectionType collection(StarlarkType elementType) {
    return new AutoValue_Types_CollectionType(elementType);
  }

  /** Returns true if {@code type} may be used as a collection. */
  public static boolean isCollection(StarlarkType type) {
    return StarlarkType.assignableFrom(COLLECTION_OF_ANY, type);
  }

  /**
   * Abstract collection type implementing common functionality. Exists to be subclassed.
   *
   * <p>{@code AbstractCollectionType}'s default {@link #assignableFromHook} always returns false if
   * {@code t} is not an rvalue-subtype of this and not of the same Java class as this. Therefore,
   * subclasses having multiple Java classes corresponding to the same Starlark type family may need
   * to override {@link #assignableFromHook}.
   */
  public abstract static class AbstractCollectionType extends StarlarkType {
    public abstract StarlarkType getElementType();

    @Override
    public boolean assignableFromHook(StarlarkType t) {
      if (t instanceof AbstractCollectionType that) {
        if (that.isRvalueAssignableTo(this)) {
          return true;
        }
        // Assume 1-1 correspondence between Java subclass and Starlark type family.
        if (this.getClass().equals(t.getClass())) {
          // Invariant in element type because `that` might be mutable.
          return StarlarkType.consistentEquals(this.getElementType(), that.getElementType());
        }
      }
      return false;
    }

    /**
     * Returns true if {@code this} is an rvalue type and is assignable to {@code that}.
     *
     * <p>Must be overridden by rvalue types.
     *
     * <p>Intended to be invoked by {@link #assignableFromHook} implementations.
     */
    // TODO: #27370 - Consider elevating to StarlarkType level if useful for non-collection types.
    protected boolean isRvalueAssignableTo(AbstractCollectionType that) {
      return false;
    }

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        // `in` and `not in` are always valid for collections on the RHS.
        case IN, NOT_IN -> thisLeft ? null : BOOL;
        default -> null;
      };
    }
  }

  /** Collection type. */
  // We need CollectionType to be a separate class from AbstractCollectionType for 2 reasons.
  // First, CollectionType is an immutable view of a collection (and so can be covariant in element
  // type), while AbstractCollectionType has mutable subtypes (which are invariant in element type).
  // Second, an @AutoValue class may not extend another - so we cannot have SequenceType or SetType
  // be subclasses of CollectionType (they are subclasses of AbstractCollectionType instead).
  @AutoValue
  public abstract static class CollectionType extends AbstractCollectionType {
    @Override
    public boolean assignableFromHook(StarlarkType t) {
      if (t instanceof AbstractCollectionType that) {
        if (that.isRvalueAssignableTo(this)) {
          return true;
        }
        // Covariant in element type when assigning from a Collection (which is immutable)
        return that instanceof CollectionType
            && StarlarkType.assignableFrom(this.getElementType(), that.getElementType());
      }
      return false;
    }

    @Override
    public final String toString() {
      return "Collection[" + getElementType() + "]";
    }

    @Override
    public CollectionType toLvalue() {
      return collection(getElementType().toLvalue());
    }
  }

  /** Sequence type */
  public static SequenceType sequence(StarlarkType elementType) {
    return new AutoValue_Types_SequenceType(elementType);
  }

  /** Abstract sequence type for common sequence functionality. Exists to be subclassed. */
  public abstract static class AbstractSequenceType extends AbstractCollectionType {
    @Override
    public abstract StarlarkType getElementType();

    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(collection(getElementType()));
    }
  }

  /** Sequence type. */
  // We need SequenceType to be a separate class from AbstractSequenceType for 2 reasons.
  // First, SequenceType is an immutable view of a sequence (and so can be covariant in element
  // type), while AbstractSequenceType has mutable subtypes (which are invariant in element type).
  // Second, an @AutoValue class may not extend another - so we cannot have ListType or TupleType
  // be subclasses of SequenceType (they are subclasses of AbstractSequenceType instead).
  @AutoValue
  public abstract static class SequenceType extends AbstractSequenceType {
    @Override
    public abstract StarlarkType getElementType();

    @Override
    public boolean assignableFromHook(StarlarkType t) {
      if (t instanceof AbstractSequenceType that) {
        if (that.isRvalueAssignableTo(this)) {
          return true;
        }
        // Covariant in element type when assigning from a Sequence (which is immutable)
        return that instanceof SequenceType
            && StarlarkType.assignableFrom(this.getElementType(), that.getElementType());
      }
      return false;
    }

    @Override
    public final String toString() {
      return "Sequence[" + getElementType() + "]";
    }

    @Override
    public SequenceType toLvalue() {
      return sequence(getElementType().toLvalue());
    }

    @Override
    protected boolean isRvalueAssignableTo(AbstractCollectionType t) {
      return false;
    }
  }

  /** Mapping type */
  public static MappingType mapping(StarlarkType keyType, StarlarkType valueType) {
    return new AutoValue_Types_MappingType(keyType, valueType);
  }

  /**
   * Abstract mapping type for common map functionality. Exists to be subclassed.
   *
   * <p>{@code AbstractMappingType}'s default {@link #assignableFromHook} always returns false if
   * {@code t} and this are not of the same Java class. Therefore, subclasses having multiple Java
   * classes corresponding to the same Starlark type family may need to override {@link
   * #assignableFromHook}.
   */
  public abstract static class AbstractMappingType extends AbstractCollectionType {
    public abstract StarlarkType getKeyType();

    public abstract StarlarkType getValueType();

    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(collection(getKeyType()));
    }

    @Override
    public StarlarkType getElementType() {
      return getKeyType();
    }

    @Override
    public boolean assignableFromHook(StarlarkType t) {
      if (t instanceof AbstractMappingType that) {
        if (that.isMappingRvalueAssignableTo(this)) {
          return true;
        }
        // Assume 1-1 correspondence between Java subclass and Starlark type family.
        if (this.getClass().equals(t.getClass())) {
          // Invariant in both key and value types because `that` might be mutable.
          return StarlarkType.consistentEquals(this.getKeyType(), that.getKeyType())
              && StarlarkType.consistentEquals(this.getValueType(), that.getValueType());
        }
      }
      return false;
    }

    @Override
    protected boolean isRvalueAssignableTo(AbstractCollectionType t) {
      return t instanceof AbstractMappingType that && this.isMappingRvalueAssignableTo(that);
    }

    /**
     * Returns true if {@code this} is an rvalue type and is assignable to {@code that}.
     *
     * <p>Must be overridden by rvalue types.
     *
     * <p>Intended to be invoked by {@link #assignableFromHook} implementations.
     */
    protected boolean isMappingRvalueAssignableTo(AbstractMappingType that) {
      return false;
    }

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType rhs, boolean thisLeft) {
      return switch (operator) {
        case PIPE ->
            // TODO: #27370 - mypy supports dict | dict, but doesn't support the | operator for
            // non-dict mappings. Should we have the same restriction? (Note that such a restriction
            // would break some uses of Bazel's native.existing_rules()).
            // TODO: #27370 - do we need to handle Neve for the key or value type?
            rhs instanceof AbstractMappingType rhsMapping
                ? dictRvalue(
                    union(getKeyType(), rhsMapping.getKeyType()),
                    union(getValueType(), rhsMapping.getValueType()))
                : null;
        default -> super.inferBinaryOperator(operator, rhs, thisLeft);
      };
    }
  }

  /** Mapping type. */
  // We need MappingType to be a separate class from AbstractMappingType for 2 reasons.
  // First, MappingType is an immutable view of a mapping (and so can be covariant in value type),
  // while AbstractMappingType has mutable subtypes (which are invariant in value type).
  // Second, an @AutoValue class may not extend another - so we cannot have DictType be a subclass
  // of MappingType (it is a subclass of AbstractMappingType instead).
  @AutoValue
  public abstract static class MappingType extends AbstractMappingType {
    @Override
    public abstract StarlarkType getKeyType();

    @Override
    public abstract StarlarkType getValueType();

    @Override
    public boolean assignableFromHook(StarlarkType t) {
      if (t instanceof AbstractMappingType that) {
        if (that.isMappingRvalueAssignableTo(this)) {
          return true;
        }
        // Invariant in key type, covariant in value type when assigning from a Mapping (which is
        // immutable).
        // TODO: #27370 - Should Mapping assignment be covariant in key type as well?
        return that instanceof MappingType
            && StarlarkType.consistentEquals(this.getKeyType(), that.getKeyType())
            && StarlarkType.assignableFrom(this.getValueType(), that.getValueType());
      }
      return false;
    }

    @Override
    public final String toString() {
      return "Mapping[" + getKeyType() + ", " + getValueType() + "]";
    }

    @Override
    public MappingType toLvalue() {
      return mapping(getKeyType().toLvalue(), getValueType().toLvalue());
    }
  }

  /** Struct type */
  public static StructType struct(ImmutableMap<String, StarlarkType> fields) {
    return new AutoValue_Types_StructType(fields);
  }

  /**
   * Struct type.
   *
   * <p>This is intended to be either the type or a supertype for values implementing {@link
   * net.starlark.java.eval.Structure} - for example, Bazel's structs and providers.
   *
   * <p>Morally non-struct types shouldn't add a {@link StructType} to their supertypes just because
   * they happen to have fields. For example, a {@code list} has {@code append} and {@code extend}
   * methods, but it is *not* a subtype of {@code struct[{"append": ..., "extend": ...}]}.
   */
  @AutoValue
  public abstract static class StructType extends StarlarkType {
    /** Returns the names and types of the mandatory fields of this struct type. */
    // TODO: #27370 - should we add optional fields? (Maybe useful for Bazel's providers.)
    // TODO: #27370 - should we add mutable fields / hasSetField()?
    public abstract ImmutableMap<String, StarlarkType> getFields();

    @Override
    public boolean assignableFromHook(StarlarkType t) {
      if (t instanceof StructType that) {
        // Covariant in LHS fields; LHS field names must be a subset of RHS field names.
        return this.getFields().entrySet().stream()
            .allMatch(
                entry1 -> {
                  String fieldName = entry1.getKey();
                  StarlarkType fieldType1 = entry1.getValue();
                  @Nullable StarlarkType fieldType2 = that.getField(fieldName);
                  return fieldType2 != null && assignableFrom(fieldType1, fieldType2);
                });
      }
      return false;
    }

    @Nullable
    @Override
    public StarlarkType getField(String name, TypeContext context) {
      return getField(name);
    }

    /**
     * Returns the type of the field with the given name, or null if there is no such field.
     *
     * <p>Unlike for {@link StarlarkType#getField}, this method doesn't take a {@link TypeContext}
     * because it's expected that the names and types of a struct's fields are fixed at type
     * construction time.
     */
    @Nullable
    public StarlarkType getField(String name) {
      return getFields().get(name);
    }

    @Override
    public final String toString() {
      StringBuilder buf = new StringBuilder();
      buf.append("struct[");
      TypeConstructor.Arg.TypeDict.print(buf, getFields());
      buf.append("]");
      return buf.toString();
    }

    @Override
    public StructType toLvalue() {
      ImmutableMap.Builder<String, StarlarkType> builder = ImmutableMap.builder();
      for (Map.Entry<String, StarlarkType> entry : getFields().entrySet()) {
        builder.put(entry.getKey(), entry.getValue().toLvalue());
      }
      return struct(builder.buildOrThrow());
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

  private static ImmutableList<StarlarkType> toStarlarkTypes(
      String name, ImmutableList<TypeConstructor.Arg> args) throws TypeConstructor.Failure {
    for (TypeConstructor.Arg arg : args) {
      if (!(arg instanceof StarlarkType)) {
        throw new TypeConstructor.Failure(
            String.format("in application to %s, got '%s', expected a type", name, arg));
      }
    }
    @SuppressWarnings("unchecked") // list is immutable and all elements verified above
    var result = (ImmutableList<StarlarkType>) (ImmutableList<?>) args;
    return result;
  }

  /**
   * Returns a new type constructor wrapping the given one-argument type factory.
   *
   * <p>The type constructor can be invoked with one argument, which is passed to the underlying
   * factory, or with zero arguments, in which case the factory is invoked with {@link #ANY}. (This
   * allows, for instance, {@code list} to be treated as syntactic sugar for {@code list[Any]}.)
   */
  static TypeConstructor wrapTypeConstructor(
      String name, Function<StarlarkType, StarlarkType> factory) {
    return args -> {
      var types = toStarlarkTypes(name, args);
      return switch (types.size()) {
        case 0 -> factory.apply(ANY);
        case 1 -> factory.apply(types.get(0));
        default -> {
          throw new TypeConstructor.Failure(
              String.format("%s[] accepts exactly 1 argument but got %d", name, types.size()));
        }
      };
    };
  }

  /**
   * Returns a new type constructor wrapping the given two-argument type factory.
   *
   * <p>The type constructor can be invoked with two arguments, which are passed to the underlying
   * factory, or with zero arguments, in which case the factory is invoked with {@link #ANY} for
   * both arguments. (This allows, for instance, {@code dict} to be treated as syntactic sugar for
   * {@code dict[Any, Any]}.)
   */
  static TypeConstructor wrapTypeConstructor(
      String name, BiFunction<StarlarkType, StarlarkType, StarlarkType> factory) {
    return args -> {
      var types = toStarlarkTypes(name, args);
      return switch (types.size()) {
        case 0 -> factory.apply(ANY, ANY);
        case 2 -> factory.apply(types.get(0), types.get(1));
        default ->
            throw new TypeConstructor.Failure(
                String.format("%s[] accepts exactly 2 arguments but got %d", name, types.size()));
      };
    };
  }

  private static TypeConstructor wrapTupleConstructor() {
    // This is a function instead of a constant, so that the order of evaluation doesn't depend on
    // the position in the class.
    return args -> {
      if (args.isEmpty()) {
        // `tuple` is equivalent to `tuple[Any, ...]`
        return homogeneousTuple(ANY);
      }
      for (int i = 0; i < args.size(); i++) {
        TypeConstructor.Arg arg = args.get(i);
        if (arg.equals(TypeConstructor.Arg.ELLIPSIS)) {
          if (i == 1 && args.size() == 2) {
            return homogeneousTuple((StarlarkType) args.getFirst());
          }
          throw new TypeConstructor.Failure(
              "in application to tuple, '...' can only appear as the second of exactly 2 arguments,"
                  + " where the first argument is a type");
        } else if (arg.equals(TypeConstructor.Arg.EMPTY_TUPLE)) {
          if (args.size() == 1) {
            return Types.EMPTY_TUPLE;
          }
          throw new TypeConstructor.Failure(
              "in application to tuple, '()' can only appear if it is the only argument");
        } else if (!(arg instanceof StarlarkType)) {
          throw new TypeConstructor.Failure(
              String.format("in application to tuple, got '%s', expected a type", arg));
        }
      }
      @SuppressWarnings("unchecked") // list is immutable and all elements verified above
      var result = (ImmutableList<StarlarkType>) (ImmutableList<?>) args;
      return tuple(result);
    };
  }

  private static final TypeConstructor wrapStructConstructor() {
    return args -> {
      if (args.size() == 1) {
        TypeConstructor.Arg arg = args.getFirst();
        if (arg instanceof TypeConstructor.Arg.TypeDict dict) {
          return struct(dict.getTypes());
        } else {
          throw new TypeConstructor.Failure(
              String.format("in application to struct, got '%s', expected a dict", arg));
        }
      } else {
        throw new TypeConstructor.Failure(
            String.format("struct[] accepts exactly 1 argument but got %d", args.size()));
      }
    };
  }
}
