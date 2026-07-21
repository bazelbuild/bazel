// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;

/**
 * A factory for creating {@link StarlarkType}s, parameterized by zero or more type arguments.
 *
 * <p>Conceptually, a type constructor corresponds to what the user informally thinks of as "a
 * type": a program symbol, like {@code list}, that can appear within a type expression. The usage
 * of a constructor in a type expression yields an actual type, like {@code list[int]}. In the case
 * of basic types like {@code None} that are not parameterized, there is both a trivial nullary type
 * constructor and an underlying singleton type, where the constructor just wraps the underlying
 * type.
 */
public interface TypeConstructor {

  /** Exception thrown when a {@link TypeConstructor} is called with invalid arguments. */
  class Failure extends Exception {
    Failure(String message) {
      super(message);
    }
  }

  /**
   * An argument to a type constructor.
   *
   * <p>Conceptually, a type term is the result of {@link TypeTagger} extracting a type
   * (sub)expression. The overall expression must yield a {@link StarlarkType} -- either
   * immediately, or upon later evaluation when values are supplied for its free variables. However,
   * its subexpressions can also yield other objects such as an ellipsis or a list/dict literal;
   * those are needed for type expressions like {@code tuple[Any, ...]}, {@code struct[{"name":
   * str}]}, or {@code Callable[[int], bool]}.
   *
   * <p>A term is said to be "open" if it contains free type variables. An open term can be
   * evaluated by supplying {@link StarlarkType} values for its type variables. The result of this
   * evaluation is a non-open term, and, in the specific case of the top-level type expression's
   * term, it will be a {@link StarlarkType}.
   *
   * <p>{@link TypeConstructor#createStarlarkType} takes a vector of *non-open* type terms as
   * arguments to produce a {@link StarlarkType}.
   */
  sealed interface Term
      permits StarlarkType,
          Term.Ellipsis,
          Term.EmptyTuple,
          Term.TypeDict,
          Term.TypeVariable,
          Term.DecomposedTypeApplication,
          Term.DecomposedUnion {
    public static final Ellipsis ELLIPSIS = new Ellipsis();
    public static final EmptyTuple EMPTY_TUPLE = new EmptyTuple();

    /**
     * Returns true if this is a {@link StarlarkType} or an open type term which evaluates to a
     * {@link StarlarkType}.
     */
    default boolean isOrEvaluatesToStarlarkType() {
      return false;
    }

    /** Returns true if this type term contains free type variables. */
    default boolean isOpen() {
      return false;
    }

    /**
     * If this is an open type term, returns the term evaluated with the given binding values.
     * Otherwise, returns itself unchanged.
     *
     * @param values the values of the type variables
     */
    default Term evaluate(ImmutableList<StarlarkType> values) throws Failure {
      return this;
    }

    /** An ellipsis type argument, {@code ...}. */
    public static final class Ellipsis implements Term {
      private Ellipsis() {}

      @Override
      public String toString() {
        return "...";
      }
    }

    /** An empty tuple type argument, {@code ()}. */
    public static final class EmptyTuple implements Term {
      private EmptyTuple() {}

      @Override
      public String toString() {
        return "()";
      }
    }

    /** A dictionary with string keys and type term values, e.g. {@code {"foo": T, "bar": U}}. */
    public static final class TypeDict implements Term {
      private final ImmutableMap<String, Term> map;
      private final boolean isOpen;

      TypeDict(ImmutableMap<String, Term> map) {
        this.map = map;
        this.isOpen = map.values().stream().anyMatch(Term::isOpen);
      }

      @Override
      public boolean isOpen() {
        return isOpen;
      }

      @Override
      public TypeDict evaluate(ImmutableList<StarlarkType> values) throws Failure {
        if (!isOpen) {
          return this;
        }
        ImmutableMap.Builder<String, Term> evaluatedMap = ImmutableMap.builder();
        for (Map.Entry<String, Term> entry : map.entrySet()) {
          evaluatedMap.put(entry.getKey(), entry.getValue().evaluate(values));
        }
        return new TypeDict(evaluatedMap.buildOrThrow());
      }

      /** Returns the underlying map if all values are StarlarkTypes, or throws otherwise. */
      @SuppressWarnings("unchecked") // by construction
      public ImmutableMap<String, StarlarkType> getTypes() throws Failure {
        if (map.values().stream().allMatch(arg -> arg instanceof StarlarkType)) {
          return (ImmutableMap<String, StarlarkType>) (ImmutableMap<String, ?>) map;
        }
        throw new Failure(String.format("expected a dict with type values, got '%s'", this));
      }

      @CanIgnoreReturnValue
      static StringBuilder print(StringBuilder buf, ImmutableMap<String, ? extends Term> map) {
        buf.append('{');
        boolean first = true;
        for (Map.Entry<String, ? extends Term> entry : map.entrySet()) {
          if (!first) {
            buf.append(", ");
          }
          NodePrinter.printStringLiteral(buf, entry.getKey());
          buf.append(": ");
          buf.append(entry.getValue());
          first = false;
        }
        buf.append('}');
        return buf;
      }

      @Override
      public String toString() {
        return print(new StringBuilder(), map).toString();
      }
    }

    /**
     * A type variable used within a type expression, e.g. {@code T} in {@code list[T]}.
     *
     * <p>Because type expressions do not bind variables, type variables are always open terms, as
     * are terms containing type variables. But in the broader AST, type variables are always bound
     * by an outside piece of syntax, indicated by the {@link Resolver.Binding} they refer to.
     */
    public static final class TypeVariable implements Term {
      private final String name;
      private final Resolver.Binding binding;
      private final int paramIndex;

      TypeVariable(Identifier id, int paramIndex) {
        this.name = id.getName();
        this.binding = checkNotNull(id.getBinding(), "identifier '%s' has no binding", id);
        this.paramIndex = paramIndex;
      }

      @Override
      public boolean isOpen() {
        return true;
      }

      @Override
      public boolean isOrEvaluatesToStarlarkType() {
        return true;
      }

      @Override
      public StarlarkType evaluate(ImmutableList<StarlarkType> values) {
        return values.get(paramIndex);
      }

      public String getName() {
        return name;
      }

      public Resolver.Binding getBinding() {
        return binding;
      }

      @Override
      public String toString() {
        return name;
      }
    }

    /** A resolved but unevaluated type application expression. */
    public static final class DecomposedTypeApplication implements Term {
      private final TypeConstructor constructor;
      private final ImmutableList<Term> args;
      private final boolean isOpen;

      DecomposedTypeApplication(TypeConstructor constructor, ImmutableList<Term> args) {
        this.constructor = constructor;
        this.args = args;
        this.isOpen = args.stream().anyMatch(Term::isOpen);
      }

      @Override
      public boolean isOpen() {
        return isOpen;
      }

      @Override
      public boolean isOrEvaluatesToStarlarkType() {
        return true;
      }

      @Override
      public Term evaluate(ImmutableList<StarlarkType> values) throws Failure {
        ImmutableList.Builder<Term> evaluatedArgs = ImmutableList.builder();
        for (Term arg : args) {
          evaluatedArgs.add(arg.evaluate(values));
        }
        return constructor.createStarlarkType(evaluatedArgs.build());
      }

      public TypeConstructor getConstructor() {
        return constructor;
      }

      public ImmutableList<Term> getArgs() {
        return args;
      }

      @Override
      public String toString() {
        return args.isEmpty()
            ? constructor.toString()
            : String.format(
                "%s[%s]", constructor, args.stream().map(Term::toString).collect(joining(", ")));
      }
    }

    /** A resolved but unevaluated type union binary expression. */
    public static final class DecomposedUnion implements Term {
      private final Term x;
      private final Term y;
      private final boolean isOpen;

      DecomposedUnion(Term x, Term y) {
        checkArgument(x.isOrEvaluatesToStarlarkType() && y.isOrEvaluatesToStarlarkType());
        this.x = x;
        this.y = y;
        this.isOpen = x.isOpen() || y.isOpen();
      }

      @Override
      public boolean isOpen() {
        return isOpen;
      }

      @Override
      public boolean isOrEvaluatesToStarlarkType() {
        return true;
      }

      @Override
      public StarlarkType evaluate(ImmutableList<StarlarkType> values) throws Failure {
        return Types.union((StarlarkType) x.evaluate(values), (StarlarkType) y.evaluate(values));
      }

      public Term getX() {
        return x;
      }

      public Term getY() {
        return y;
      }

      @Override
      public String toString() {
        // Keep consistent with Types.UnionType.toString()
        return String.format("%s|%s", x, y);
      }
    }
  }

  /** A type constructor which evaluates a type expression with free variables. */
  public static final class Composite implements TypeConstructor {
    private final String name;
    private final int arity;
    private final Term root;

    Composite(String name, int arity, Term root) {
      checkArgument(root.isOrEvaluatesToStarlarkType(), "cannot evaluate %s", root);
      this.name = name;
      this.arity = arity;
      this.root = root;
    }

    @Override
    public StarlarkType createStarlarkType(ImmutableList<Term> argsTuple) throws Failure {
      if (argsTuple.size() != arity) {
        throw new Failure(
            String.format(
                "%s[] accepts exactly %s argument%s but got %d",
                name, arity, arity == 1 ? "" : "s", argsTuple.size()));
      }
      // TODO: #27370 - Should we relax the requirement that all args are StarlarkTypes?
      ImmutableList<StarlarkType> values = Types.toStarlarkTypes(name, argsTuple);
      return (StarlarkType) root.evaluate(values);
    }
  }

  /**
   * Returns the result of applying this constructor to the given type arguments, which cannot be
   * open terms.
   *
   * @throws Failure if the usage of this constructor is invalid (typically due to a mismatch in the
   *     number or type of arguments)
   */
  StarlarkType createStarlarkType(ImmutableList<Term> argsTuple) throws Failure;

  /** A type constructor that can be invoked without type arguments. */
  public interface AllowingNullary extends TypeConstructor {
    default StarlarkType createStarlarkType() {
      try {
        return createStarlarkType(ImmutableList.of());
      } catch (Failure e) {
        throw new IllegalStateException(String.format("Not nullary: %s", this), e);
      }
    }
  }
}
