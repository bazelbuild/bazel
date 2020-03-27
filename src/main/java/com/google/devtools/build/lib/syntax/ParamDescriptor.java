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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/** A value class for storing {@link Param} metadata to avoid using Java proxies. */
final class ParamDescriptor {

  private final String name;
  @Nullable private final Object defaultValue;
  private final Class<?> type;
  private final Class<?> generic1;
  private final boolean noneable;
  private final boolean named;
  private final boolean positional;
  // While the type can be inferred completely by the Param annotation, this tuple allows for the
  // type of a given parameter to be determined only once, as it is an expensive operation.
  private final SkylarkType skylarkType;
  // The semantics flag responsible for disabling this parameter, or null if enabled.
  // It is an error for Starlark code to supply a value to a disabled parameter.
  @Nullable private final String disabledByFlag;

  private ParamDescriptor(
      String name,
      String defaultExpr,
      Class<?> type,
      Class<?> generic1,
      boolean noneable,
      boolean named,
      boolean positional,
      SkylarkType skylarkType,
      @Nullable String disabledByFlag) {
    this.name = name;
    this.defaultValue = defaultExpr.isEmpty() ? null : evalDefault(name, defaultExpr);
    this.type = type;
    this.generic1 = generic1;
    this.noneable = noneable;
    this.named = named;
    this.positional = positional;
    this.skylarkType = skylarkType;
    this.disabledByFlag = disabledByFlag;
  }

  /**
   * Returns a {@link ParamDescriptor} representing the given raw {@link Param} annotation and the
   * given semantics.
   */
  static ParamDescriptor of(Param param, StarlarkSemantics starlarkSemantics) {
    Class<?> type = param.type();
    Class<?> generic = param.generic1();
    boolean noneable = param.noneable();

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

    return new ParamDescriptor(
        param.name(),
        defaultExpr,
        type,
        generic,
        noneable,
        param.named(),
        param.positional(),
        getType(type, generic, param.allowedTypes(), noneable),
        disabledByFlag);
  }

  /** @see Param#name() */
  String getName() {
    return name;
  }

  /** @see Param#type() */
  Class<?> getType() {
    return type;
  }

  private static SkylarkType getType(
      Class<?> type, Class<?> generic, ParamType[] allowedTypes, boolean noneable) {
    SkylarkType result = SkylarkType.BOTTOM;
    if (allowedTypes.length > 0) {
      Preconditions.checkState(Object.class.equals(type));
      for (ParamType paramType : allowedTypes) {
        Class<?> generic1 = paramType.generic1();
        SkylarkType t =
            generic1 != Object.class
                ? SkylarkType.of(paramType.type(), generic1)
                : SkylarkType.of(paramType.type());
        result = SkylarkType.Union.of(result, t);
      }
    } else {
      result = generic != Object.class ? SkylarkType.of(type, generic) : SkylarkType.of(type);
    }

    if (noneable) {
      result = SkylarkType.Union.of(result, SkylarkType.NONE);
    }
    return result;
  }

  /** @see Param#generic1() */
  Class<?> getGeneric1() {
    return generic1;
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

  SkylarkType getSkylarkType() {
    return skylarkType;
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
    // Common cases; also needed for bootstrapping UNIVERSE.
    if (expr.equals("None")) {
      return Starlark.NONE;
    } else if (expr.equals("True")) {
      return true;
    } else if (expr.equals("False")) {
      return false;
    } else if (expr.equals("unbound")) {
      return Starlark.UNBOUND;
    }

    Object x = defaultValueCache.get(expr);
    if (x != null) {
      return x;
    }
    try (Mutability mutability = Mutability.create("initialization")) {
      // Note that this Starlark thread ignores command line flags.
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .useDefaultSemantics()
              .setGlobals(Module.createForBuiltins(Starlark.UNIVERSE))
              .build();
      Module module = thread.getGlobals();

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

      x = EvalUtils.eval(ParserInput.fromLines(expr), FileOptions.DEFAULT, module, thread);
    } catch (InterruptedException ex) {
      throw new IllegalStateException(ex); // can't happen
    } catch (SyntaxError | EvalException ex) {
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
