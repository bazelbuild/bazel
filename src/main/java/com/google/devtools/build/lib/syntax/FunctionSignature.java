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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Function Signatures for BUILD language (same as Python)
 *
 * <p>Starlark's function signatures are just like Python3's. A function may have 6 kinds of
 * parameters: positional mandatory, positional optional, positional rest (aka *args or variadic
 * parameter), keyword-only mandatory, keyword-only optional, keyword rest (aka **kwargs parameter).
 * A caller may specify all arguments but the *args and **kwargs arguments by name, and thus all
 * mandatory and optional parameters are named parameters.
 *
 * <p>To enable various optimizations in the argument processing routine, we sort parameters
 * according the following constraints, enabling corresponding optimizations:
 *
 * <ol>
 *   <li>The positional mandatories come just before the positional optionals, so they can be filled
 *       in one go.
 *   <li>Positionals come first, so it's easy to prepend extra positional arguments such as "self"
 *       to an argument list, and we optimize for the common case of no key-only mandatory
 *       parameters. key-only parameters are thus grouped together. positional mandatory and
 *       key-only mandatory parameters are separate, but there is no loop over a contiguous chunk of
 *       them, anyway.
 *   <li>The named are all grouped together, with star and star_star rest parameters coming last.
 *   <li>Mandatory parameters in each category (positional and named-only) come before the optional
 *       parameters, for the sake of slightly better clarity to human implementers. This eschews an
 *       optimization whereby grouping optionals together allows to iterate over them in one go
 *       instead of two; however, this relatively minor optimization only matters when keyword
 *       arguments are passed, at which point it is dwarfed by the slowness of keyword processing.
 * </ol>
 *
 * <p>Parameters are thus sorted in the following order: positional mandatory parameters (if any),
 * positional optional parameters (if any), key-only mandatory parameters (if any), key-only
 * optional parameters (if any), then star parameter (if any), then star_star parameter (if any).
 */
@AutoCodec
@AutoValue
public abstract class FunctionSignature {

  // These abstract getters specify the actual parameter count fields to be defined by AutoValue.

  /** Number of mandatory positional parameters */
  public abstract int numMandatoryPositionals();

  /** Number of optional positional parameters */
  public abstract int numOptionalPositionals();

  /** Number of mandatory named-only parameters. */
  public abstract int numMandatoryNamedOnly();

  /** Number of optional named-only parameters */
  public abstract int numOptionalNamedOnly();

  /** True if function has variadic parameter, {@code def f(*args)}. */
  public abstract boolean hasVarargs();

  /** True if function has residual keyword-argument parameter, {@code def f(**kwargs)}. */
  public abstract boolean hasKwargs();

  /** Parameter names. */
  public abstract ImmutableList<String> getParameterNames();

  // computed parameter counts

  /** Number of optional and mandatory positional parameters. */
  public int numPositionals() {
    return numMandatoryPositionals() + numOptionalPositionals();
  }

  /** Number of optional and mandatory named-only parameters. */
  public int numNamedOnly() {
    return numMandatoryNamedOnly() + numOptionalNamedOnly();
  }

  /** number of optional parameters. */
  public int numOptionals() {
    return numOptionalPositionals() + numOptionalNamedOnly();
  }

  private boolean hasStar() {
    return hasVarargs() || (numNamedOnly() > 0);
  }

  /** total number of parameters */
  public int numParameters() {
    return numPositionals() + numNamedOnly() + (hasStar() ? 1 : 0) + (hasKwargs() ? 1 : 0);
  }

  private static final Interner<ImmutableList<String>> namesInterner =
      BlazeInterners.newWeakInterner();

  /** Intern a list of names. */
  private static ImmutableList<String> names(List<String> names) {
    return namesInterner.intern(
        names.stream().map(StringCanonicalizer::intern).collect(toImmutableList()));
  }

  // Interner.
  // Are there really a significant number of duplicates? Why??
  private static final Interner<FunctionSignature> signatureInterner =
      BlazeInterners.newWeakInterner();

  // TODO(adonovan): not a user-friendly API. Provide external callers with this function:
  //   FunctionSignature.parse("a, b=1, *, c, d=2, *args, **kwargs")
  // implemented by invoking the Starlark parser. (Most uses are in tests.)
  @AutoCodec.Instantiator
  public static FunctionSignature create(
      int numMandatoryPositionals,
      int numOptionalPositionals,
      int numMandatoryNamedOnly,
      int numOptionalNamedOnly,
      boolean hasVarargs,
      boolean hasKwargs,
      ImmutableList<String> parameterNames) {
    Preconditions.checkArgument(
        0 <= numMandatoryPositionals
            && 0 <= numOptionalPositionals
            && 0 <= numMandatoryNamedOnly
            && 0 <= numOptionalNamedOnly);

    FunctionSignature sig =
        new AutoValue_FunctionSignature(
            numMandatoryPositionals,
            numOptionalPositionals,
            numMandatoryNamedOnly,
            numOptionalNamedOnly,
            hasVarargs,
            hasKwargs,
            names(parameterNames));

    return signatureInterner.intern(sig);
  }

  @Override
  public final String toString() {
    StringBuilder sb = new StringBuilder();
    toStringBuilder(sb, null); // no default values
    return sb.toString();
  }

  // ElementPrinter returns the string form of the ith element of a sequence.
  interface ElementPrinter {
    String print(int i);
  }

  /**
   * Appends a representation of this signature to a string buffer.
   *
   * @param printer output StringBuilder
   * @param defaultValuePrinter optional callback for formatting i'th default value (if any).
   */
  StringBuilder toStringBuilder(
      final StringBuilder printer, @Nullable final ElementPrinter defaultValuePrinter) {
    final ImmutableList<String> names = getParameterNames();

    int mandatoryPositionals = numMandatoryPositionals();
    int optionalPositionals = numOptionalPositionals();
    int mandatoryNamedOnly = numMandatoryNamedOnly();
    int optionalNamedOnly = numOptionalNamedOnly();
    boolean hasVarargs = hasVarargs();
    boolean hasKwargs = hasKwargs();
    int positionals = mandatoryPositionals + optionalPositionals;
    int namedOnly = mandatoryNamedOnly + optionalNamedOnly;
    int named = positionals + namedOnly;
    int args = named + (hasVarargs ? 1 : 0) + (hasKwargs ? 1 : 0);
    int endMandatoryNamedOnly = positionals + mandatoryNamedOnly;
    int iStarArg = named;
    int iKwArg = args - 1;

    class Show {
      private boolean isMore = false;
      private int j = 0;

      public void comma() {
        if (isMore) {
          printer.append(", ");
        }
        isMore = true;
      }

      public void mandatory(int i) {
        comma();
        printer.append(names.get(i));
      }

      public void optional(int i) {
        mandatory(i);
        if (defaultValuePrinter != null) {
          String str = defaultValuePrinter.print(j++);
          printer.append(" = ").append(str != null ? str : "?");
        }
      }
    }

    Show show = new Show();

    int i = 0;
    for (; i < mandatoryPositionals; i++) {
      show.mandatory(i);
    }
    for (; i < positionals; i++) {
      show.optional(i);
    }
    if (hasStar()) {
      show.comma();
      printer.append("*");
      if (hasVarargs) {
        printer.append(names.get(iStarArg));
      }
    }
    for (; i < endMandatoryNamedOnly; i++) {
      show.mandatory(i);
    }
    for (; i < named; i++) {
      show.optional(i);
    }
    if (hasKwargs) {
      show.comma();
      printer.append("**");
      printer.append(names.get(iKwArg));
    }

    return printer;
  }

  /** Convert a list of Parameter into a FunctionSignature. */
  static FunctionSignature fromParameters(Iterable<Parameter> parameters)
      throws SignatureException {
    int mandatoryPositionals = 0;
    int optionalPositionals = 0;
    int mandatoryNamedOnly = 0;
    int optionalNamedOnly = 0;
    boolean hasStarStar = false;
    boolean hasStar = false;
    @Nullable String star = null;
    @Nullable String starStar = null;
    ArrayList<String> params = new ArrayList<>();
    // optional named-only parameters are kept aside to be spliced after the mandatory ones.
    ArrayList<String> optionalNamedOnlyParams = new ArrayList<>();
    boolean defaultRequired = false; // true after mandatory positionals and before star.
    Set<String> paramNameSet = new HashSet<>(); // set of names, to avoid duplicates

    for (Parameter param : parameters) {
      if (hasStarStar) {
        throw new SignatureException("illegal parameter after star-star parameter", param);
      }
      @Nullable String name = param.getName();
      if (param.getName() != null) {
        if (paramNameSet.contains(name)) {
          throw new SignatureException("duplicate parameter name in function definition", param);
        }
        paramNameSet.add(name);
      }
      if (param instanceof Parameter.StarStar) {
        hasStarStar = true;
        starStar = name;
      } else if (param instanceof Parameter.Star) {
        if (hasStar) {
          throw new SignatureException("duplicate star parameter in function definition", param);
        }
        hasStar = true;
        defaultRequired = false;
        if (param.getName() != null) {
          star = name;
        }
      } else if (hasStar && param instanceof Parameter.Optional) {
        optionalNamedOnly++;
        optionalNamedOnlyParams.add(name);
      } else {
        params.add(name);
        if (param instanceof Parameter.Optional) {
          optionalPositionals++;
          defaultRequired = true;
        } else if (hasStar) {
          mandatoryNamedOnly++;
        } else if (defaultRequired) {
          throw new SignatureException(
              "a mandatory positional parameter must not follow an optional parameter", param);
        } else {
          mandatoryPositionals++;
        }
      }
    }
    params.addAll(optionalNamedOnlyParams);

    if (star != null) {
      params.add(star);
    }
    if (starStar != null) {
      params.add(starStar);
    }
    return FunctionSignature.create(
        mandatoryPositionals,
        optionalPositionals,
        mandatoryNamedOnly,
        optionalNamedOnly,
        star != null,
        starStar != null,
        ImmutableList.copyOf(params));
  }

  /**
   * Constructs a function signature (with names) from signature description and names. This method
   * covers the general case. The number of optional named-only parameters is deduced from the other
   * arguments.
   *
   * @param numMandatoryPositionals an int for the number of mandatory positional parameters
   * @param numOptionalPositionals an int for the number of optional positional parameters
   * @param numMandatoryNamedOnly an int for the number of mandatory named-only parameters
   * @param hasVarargs whether function is variadic parameter
   * @param hasKwargs whether function accepts arbitrary named arguments
   * @param names an Array of String for the parameter names
   */
  static FunctionSignature of(
      int numMandatoryPositionals,
      int numOptionalPositionals,
      int numMandatoryNamedOnly,
      boolean hasVarargs,
      boolean hasKwargs,
      String... names) {
    return create(
        numMandatoryPositionals,
        numOptionalPositionals,
        numMandatoryNamedOnly,
        names.length
            - (hasKwargs ? 1 : 0)
            - (hasVarargs ? 1 : 0)
            - numMandatoryPositionals
            - numOptionalPositionals
            - numMandatoryNamedOnly,
        hasVarargs,
        hasKwargs,
        ImmutableList.copyOf(names));
  }

  /**
   * Constructs a function signature from mandatory positional argument names.
   *
   * @param names an Array of String for the positional parameter names
   * @return a FunctionSignature
   */
  public static FunctionSignature of(String... names) {
    return of(names.length, 0, 0, false, false, names);
  }

  /**
   * Constructs a function signature from positional argument names.
   *
   * @param numMandatory an int for the number of mandatory positional parameters
   * @param names an Array of String for the positional parameter names
   * @return a FunctionSignature
   */
  public static FunctionSignature of(int numMandatory, String... names) {
    return of(numMandatory, names.length - numMandatory, 0, false, false, names);
  }

  /**
   * Constructs a function signature from named-only parameter names.
   *
   * @param numMandatory an int for the number of mandatory named-only parameters
   * @param names an Array of String for the named-only parameter names
   * @return a FunctionSignature
   */
  public static FunctionSignature namedOnly(int numMandatory, String... names) {
    return of(0, 0, numMandatory, false, false, names);
  }

  /** Invalid signature from Parser or from SkylarkCallable annotation. */
  static class SignatureException extends Exception {
    private final Parameter parameter;

    /** SignatureException from a message and a Parameter */
    SignatureException(String message, Parameter parameter) {
      super(message);
      this.parameter = parameter;
    }

    /** Returns the parameter that caused the exception. */
    Parameter getParameter() {
      return parameter;
    }
  }

  /** A ready-made signature to allow only keyword parameters and put them in a kwarg parameter */
  public static final FunctionSignature KWARGS = of(0, 0, 0, false, true, "kwargs");

  /** A ready-made signature that accepts no arguments. */
  public static final FunctionSignature NOARGS = of(0, 0, 0, false, false);

  /** A ready-made signature that allows any arguments. */
  public static final FunctionSignature ANY = of(0, 0, 0, true, true, "args", "kwargs");
}
