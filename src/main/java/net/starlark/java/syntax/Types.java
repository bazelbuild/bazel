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
import com.google.common.collect.ImmutableCollection;
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

  // A frequently-used union `int | float`.
  public static final UnionType NUMERIC = (UnionType) union(INT, FLOAT);

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

    @Override
    public StarlarkType getField(String name) {
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

    @Override
    protected boolean isComparable(StarlarkType that) {
      // Regard Never - as the bottom type - to be comparable to anything; in particular, this
      // allows empty lists (i.e. list[Never]) to be comparable to arbitrary non-empty lists.
      return true;
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

    @Override
    protected boolean isComparable(StarlarkType that) {
      return that.equals(Types.BOOL) || that.equals(Types.ANY);
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
      // TODO: #27370 - we are expressing "other is-consistent-subtype-of NUMERIC", which supposed
      // to be (but currently isn't) implemented by StarlarkType.assignableFrom.
      return that.equals(INT) || that.equals(FLOAT) || that.equals(NUMERIC) || that.equals(ANY);
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
      // TODO: #27370 - we are expressing "other is-consistent-subtype-of NUMERIC", which supposed
      // to be (but currently isn't) implemented by StarlarkType.assignableFrom.
      return that.equals(INT) || that.equals(FLOAT) || that.equals(NUMERIC) || that.equals(ANY);
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
    protected boolean isComparable(StarlarkType that) {
      return getTypes().stream().allMatch(type -> StarlarkType.comparable(type, that));
    }
  }

  public static ListType list(StarlarkType elementType) {
    return new AutoValue_Types_ListType(elementType);
  }

  /** List type */
  @AutoValue
  public abstract static class ListType extends AbstractSequenceType {
    @Override
    public List<StarlarkType> getSupertypes() {
      return ImmutableList.of(sequence(getElementType()), collection(getElementType()));
    }

    @Override
    public final String toString() {
      return "list[" + getElementType() + "]";
    }

    @Override
    @Nullable
    StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
      return switch (operator) {
        case PLUS ->
            that instanceof ListType thatList
                ? list(union(getElementType(), thatList.getElementType()))
                : null;
        case STAR -> that.equals(Types.INT) ? this : null;
        default -> super.inferBinaryOperator(operator, that, thisLeft);
      };
    }

    @Override
    protected boolean isComparable(StarlarkType that) {
      if (that.equals(Types.ANY)) {
        return true;
      } else if (that instanceof ListType thatList) {
        return comparable(getElementType(), thatList.getElementType());
      }
      return false;
    }
  }

  public static DictType dict(StarlarkType keyType, StarlarkType valueType) {
    return new AutoValue_Types_DictType(keyType, valueType);
  }

  /** Dict type */
  @AutoValue
  public abstract static class DictType extends AbstractMappingType {
    @Override
    public abstract StarlarkType getKeyType();

    @Override
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
  }

  public static TupleType tuple(ImmutableList<StarlarkType> elementTypes) {
    return new AutoValue_Types_TupleType(elementTypes);
  }

  /** Tuple type of a fixed length. */
  @AutoValue
  public abstract static class TupleType extends AbstractSequenceType {
    public abstract ImmutableList<StarlarkType> getElementTypes();

    @Override
    public StarlarkType getElementType() {
      return union(getElementTypes());
    }

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

    /** Returns the type of this tuple concatenated with another. */
    TupleType concatenate(TupleType rhs) {
      // TODO: #27728 - revisit concatenation when we support tuples of indeterminate shape.
      return tuple(
          ImmutableList.<StarlarkType>builder()
              .addAll(getElementTypes())
              .addAll(rhs.getElementTypes())
              .build());
    }

    /** Returns the type of this tuple repeated. */
    TupleType repeat(int times) {
      // TODO: #27728 - revisit concatenation when we support tuples of indeterminate shape.
      ImmutableList.Builder<StarlarkType> builder = ImmutableList.builder();
      for (int i = 0; i < times; i++) {
        builder.addAll(getElementTypes());
      }
      return tuple(builder.build());
    }

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

    @Override
    protected boolean isComparable(StarlarkType that) {
      if (that.equals(Types.ANY)) {
        return true;
      } else if (that instanceof TupleType thatTuple) {
        int commonLength = Math.min(getElementTypes().size(), thatTuple.getElementTypes().size());
        for (int i = 0; i < commonLength; i++) {
          if (!comparable(getElementTypes().get(i), thatTuple.getElementTypes().get(i))) {
            return false;
          }
        }
        return true;
      }
      return false;
    }
  }

  /** Collection type */
  public static CollectionType collection(StarlarkType elementType) {
    return new AutoValue_Types_CollectionType(elementType);
  }

  /** Abstract collection type implementing common functionality. Exists to be subclassed. */
  public abstract static class AbstractCollectionType extends StarlarkType {
    public abstract StarlarkType getElementType();

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
  // We need CollectionType to be a separate class from AbstractCollectionType only because one
  // @AutoValue class may not extend another - so we cannot have SequenceType or SetType be
  // subclasses of CollectionType (they are subclasses of AbstractCollectionType instead).
  @AutoValue
  public abstract static class CollectionType extends AbstractCollectionType {
    @Override
    public final String toString() {
      return "Collection[" + getElementType() + "]";
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
  // We need SequenceType to be a separate class from AbstractSequenceType only because one
  // @AutoValue class may not extend another - so we cannot have ListType or
  // TupleType be subclasses of SequenceType (they are subclasses of AbstractSequenceType instead).
  @AutoValue
  public abstract static class SequenceType extends AbstractSequenceType {
    @Override
    public abstract StarlarkType getElementType();

    @Override
    public final String toString() {
      return "Sequence[" + getElementType() + "]";
    }
  }

  /** Mapping type */
  public static MappingType mapping(StarlarkType keyType, StarlarkType valueType) {
    return new AutoValue_Types_MappingType(keyType, valueType);
  }

  /** Abstract mapping type for common map functionality. Exists to be subclassed. */
  public abstract static class AbstractMappingType extends AbstractCollectionType {
    public abstract StarlarkType getKeyType();

    public abstract StarlarkType getValueType();

    @Override
    public StarlarkType getElementType() {
      return getKeyType();
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
                ? dict(
                    union(getKeyType(), rhsMapping.getKeyType()),
                    union(getValueType(), rhsMapping.getValueType()))
                : null;
        default -> super.inferBinaryOperator(operator, rhs, thisLeft);
      };
    }
  }

  /** Mapping type. */
  // We need MappingType to be a separate class from AbstractMappingType only because one @AutoValue
  // class may not extend another - so we cannot have DictType be a subclass of MappingType (it is a
  // subclass of AbstractMappingType instead).
  @AutoValue
  public abstract static class MappingType extends AbstractMappingType {
    @Override
    public abstract StarlarkType getKeyType();

    @Override
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

  private static final TypeConstructor wrapTupleConstructor() {
    // This is a function instead of a constant, so that the order of evaluation doesn't depend on
    // the position in the class.
    return args -> tuple(toStarlarkTypes("tuple", args));
  }
}
