package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Set;
import java.util.TreeSet;

/**
 * A set of Validator functions, either Starlark or native allows for validation of Package just
 * before it is built. This allows for RuleClasses in a BUILD file to veto the creation of package
 * that violates its usage rules. This is primarily useful when those rules can only be applied by
 * inspecting the whole package.
 */
final class PackageValidators {
  private static final Comparator<StarlarkFunction> BASE_FUNC_COMP =
      Comparator.<StarlarkFunction, String>comparing(bf -> bf.getName())
          .thenComparing(bf -> bf.getLocation().toString());

  private static final Comparator<PackageValidator> NATIVE_VALIDATOR_COMP =
      Comparator.<PackageValidator, String>comparing(v -> v.getClass().getName());

  private final Set<StarlarkFunction> skylarkValidators;
  private final Set<PackageValidator> nativeValidators;

  /** Noop validators. */
  PackageValidators() {
    skylarkValidators = ImmutableSet.of();
    nativeValidators = ImmutableSet.of();
  }

  PackageValidators(ImmutableList<PackageValidator> globalValidators) {
    skylarkValidators = new TreeSet<>(BASE_FUNC_COMP);
    nativeValidators = new TreeSet<>(NATIVE_VALIDATOR_COMP);
    nativeValidators.addAll(globalValidators);
  }

  void addSkylarkValidator(StarlarkFunction v) {
    skylarkValidators.add(v);
  }

  void addNativeValidator(PackageValidator v) {
    nativeValidators.add(v);
  }

  public void validate(Package.Builder pkgBuilder, StarlarkThread thread) {
    if (skylarkValidators.isEmpty() && nativeValidators.isEmpty()) {
      return;
    }
    // Starlark first, passing args = (String, dict()) to the validator
    // function.
    try {
      ArrayList<Object> args = new ArrayList<>();
      Dict<String, Dict<String, Object>> pkg = SkylarkNativeModule.packageDict(thread);
      args.add(pkgBuilder.getBuildFileLabel().getPackageName());
      args.add(pkg);
      for (StarlarkFunction v : skylarkValidators) {
        Starlark.call(thread, v, args, ImmutableMap.of());
      }
    } catch (EvalException e) {
      pkgBuilder.addEvent(Event.error(e.getLocation(), e.getMessage()));
      pkgBuilder.setContainsErrors();
    } catch (InterruptedException e) {
      pkgBuilder.addEvent(Event.error(Location.BUILTIN, e.getMessage()));
      pkgBuilder.setContainsErrors();
    }

    // Then native PackageValidator interfaces.
    ImmutableSet<Target> targets = ImmutableSet.copyOf(pkgBuilder.getTargets());
    try {
      for (PackageValidator v : nativeValidators) {
        v.validate(pkgBuilder.getBuildFileLabel().getPackageName(), targets);
      }
    } catch (Exception e) {
      pkgBuilder.addEvent(Event.error(Location.BUILTIN, e.getMessage()));
      pkgBuilder.setContainsErrors();
    }
  }

  @Override
  public String toString() {
    return "PackageValidator with validators nat=" + nativeValidators + " str=" + skylarkValidators;
  }
}
