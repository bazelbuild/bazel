package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableSet;

/** Allows a Native Rule to validate the contents of a Package before creation */
public interface PackageValidator {
  /**
   * Give a RuleClass a chance to validate the given Set of Targets, for the given Package. If a
   * violation of this PackageValidator's expection occurs an Exception should be, which will be
   * turned into a builtin Bazel error on the package.
   */
  default void validate(String packageName, ImmutableSet<Target> targets) {}
}
