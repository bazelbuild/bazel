// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.SyntaxError;

/** A value class for storing {@link Param} metadata to avoid using Java proxies. */
final class ParamDescriptor {

  private final String name;
  @Nullable private final Object defaultValue;
  private final boolean noneable;
  private final boolean named;
  private final boolean positional;
  private final List<Class<?>> allowedClasses; // non-empty
  // The semantics flag responsible for disabling this parameter, or null if enabled.
  // It is an error for Starlark code to supply a value to a disabled parameter.
  @Nullable private final String disabledByFlag;

  private ParamDescriptor(
      String name,
      String defaultExpr,
      boolean noneable,
      boolean named,
      boolean positional,
      List<Class<?>> allowedClasses,
      @Nullable String disabledByFlag) {
    this.name = name;
    this.defaultValue = defaultExpr.isEmpty() ? null : evalDefault(name, defaultExpr);
    this.noneable = noneable;
    this.named = named;
    this.positional = positional;
    this.allowedClasses = allowedClasses;
    this.disabledByFlag = disabledByFlag;
  }

  /**
   * Returns a {@link ParamDescriptor} representing the given raw {@link Param} annotation and the
   * given semantics.
   */
  static ParamDescriptor of(Param param, Class<?> paramClass, StarlarkSemantics starlarkSemantics) {
    String defaultExpr = param.defaultValue();
    String disabledByFlag = null;
    if (!starlarkSemantics.isFeatureEnabledBasedOnTogglingFlags(
        param.enableOnlyWithFlag(), param.disableWithFlag())) {
      defaultExpr = param.valueWhenDisabled();
      disabledByFlag =
          !param.enableOnlyWithFlag().isEmpty()
              ? param.enableOnlyWithFlag()
              : param.disableWithFlag();
      Preconditions.checkState(!disabledByFlag.isEmpty());
    }

    // Compute set of allowed classes.
    ParamType[] allowedTypes = param.allowedTypes();
    List<Class<?>> allowedClasses = new ArrayList<>();
    if (allowedTypes.length > 0) {
      for (ParamType pt : allowedTypes) {
        allowedClasses.add(pt.type());
      }
    } else if (param.type() == Void.class) {
      // If no Param.type type was specified, use the class of the parameter itself.
      // Interpret primitive boolean parameter as j.l.Boolean.
      allowedClasses.add(paramClass == Boolean.TYPE ? Boolean.class : paramClass);
    } else {
      allowedClasses.add(param.type());
    }
    if (param.noneable()) {
      // A few annotations redundantly declare NoneType.
      if (!allowedClasses.contains(NoneType.class)) {
        allowedClasses.add(NoneType.class);
      }
    }

    return new ParamDescriptor(
        param.name(),
        defaultExpr,
        param.noneable(),
        param.named(),
        param.positional(),
        allowedClasses,
        disabledByFlag);
  }

  /** @see Param#name() */
  String getName() {
    return name;
  }

  /** Returns a description of allowed argument types suitable for an error message. */
  String getTypeErrorMessage() {
    return allowedClasses.stream().map(Starlark::classType).collect(Collectors.joining(" or "));
  }

  List<Class<?>> getAllowedClasses() {
    return allowedClasses;
  }

  /** @see Param#noneable() */
  boolean isNoneable() {
    return noneable;
  }

  /** @see Param#positional() */
  boolean isPositional() {
    return positional;
  }

  /** @see Param#named() */
  boolean isNamed() {
    return named;
  }

  /** Returns the effective default value of this parameter, or null if mandatory. */
  @Nullable
  Object getDefaultValue() {
    return defaultValue;
  }

  /** Returns the flag responsible for disabling this parameter, or null if it is enabled. */
  @Nullable
  String disabledByFlag() {
    return disabledByFlag;
  }

  // A memoization of evalDefault, keyed by expression.
  // This cache is manually maintained (instead of using LoadingCache),
  // as default values may sometimes be recursively requested.
  private static final ConcurrentHashMap<String, Object> defaultValueCache =
      new ConcurrentHashMap<>();

  // Evaluates the default value expression for a parameter.
  private static Object evalDefault(String name, String expr) {
    // Values required by defaults of functions in UNIVERSE must
    // be handled without depending on the evaluator, or even
    // on defaultValueCache, because JVM global variable initialization
    // is such a mess. (Specifically, it's completely dynamic,
    // so if two or more variables are mutually dependent, like
    // defaultValueCache and UNIVERSE would be, you have to write
    // code that works in all possible dynamic initialization orders.)
    // Better not to go there.
    if (expr.equals("None")) {
      return Starlark.NONE;
    } else if (expr.equals("True")) {
      return true;
    } else if (expr.equals("False")) {
      return false;
    } else if (expr.equals("unbound")) {
      return Starlark.UNBOUND;
    } else if (expr.equals("0")) {
      return StarlarkInt.of(0);
    } else if (expr.equals("1")) {
      return StarlarkInt.of(1);
    } else if (expr.equals("[]")) {
      return StarlarkList.empty();
    } else if (expr.equals("()")) {
      return Tuple.empty();
    } else if (expr.equals("\" \"")) {
      return " ";
    }

    Object x = defaultValueCache.get(expr);
    if (x != null) {
      return x;
    }

    // We can't evaluate Starlark code until UNIVERSE is bootstrapped.
    if (Starlark.UNIVERSE == null) {
      throw new IllegalStateException("no bootstrap value for " + name + "=" + expr);
    }

    Module module = Module.create();
    try (Mutability mu = Mutability.create("Builtin param default init")) {
      // Note that this Starlark thread ignores command line flags.
      StarlarkThread thread = new StarlarkThread(mu, StarlarkSemantics.DEFAULT);

      // Disable polling of the java.lang.Thread.interrupt flag during
      // Starlark evaluation. Assuming the expression does not call a
      // built-in that throws InterruptedException, this allows us to
      // assert that InterruptedException "can't happen".
      //
      // Bazel Java threads are routinely interrupted during Starlark execution,
      // and the Starlark interpreter may be in a call to LoadingCache (in CallUtils).
      // LoadingCache computes the cache entry in the same thread that first
      // requested the entry, propagating undesirable thread state (which Einstein
      // called "spooky action at a distance") from an arbitrary application thread
      // to here, which is logically one-time initialization code.
      //
      // A simpler non-solution would be to use a "clean" pool thread
      // to compute each cache entry; we could safely assume such a thread
      // is never interrupted. However, this runs afoul of JVM class initialization:
      // the initialization of Starlark.UNIVERSE depends on Starlark.UNBOUND
      // because of the reference above. That's fine if they are initialized by
      // the same thread, as JVM class initialization locks are reentrant,
      // but the reference deadlocks if made from another thread.
      // See https://docs.oracle.com/javase/specs/jls/se12/html/jls-12.html#jls-12.4
      thread.ignoreThreadInterrupts();

      x = Starlark.eval(ParserInput.fromLines(expr), FileOptions.DEFAULT, module, thread);
    } catch (InterruptedException ex) {
      throw new IllegalStateException(ex); // can't happen
    } catch (SyntaxError.Exception | EvalException ex) {
      throw new IllegalArgumentException(
          String.format(
              "failed to evaluate default value '%s' of parameter '%s': %s",
              expr, name, ex.getMessage()),
          ex);
    }
    defaultValueCache.put(expr, x);
    return x;
  }
}
