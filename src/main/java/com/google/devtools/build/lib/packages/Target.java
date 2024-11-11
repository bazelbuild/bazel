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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.License.DistributionType;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A node in the build dependency graph, identified by a Label.
 *
 * <p>This StarlarkBuiltin does not contain any documentation since Starlark's Target type refers to
 * TransitiveInfoCollection.class, which contains the appropriate documentation.
 */
public interface Target extends TargetData {

  /** Returns the Package to which this target belongs. */
  Package getPackage();

  /**
   * Returns the innermost symbolic macro that declared this target, or null if it was declared
   * outside any symbolic macro (i.e. directly in a BUILD file or only in one or more legacy
   * macros).
   *
   * <p>For targets in deserialized packages, throws {@link IllegalStateException}.
   */
  @Nullable
  default MacroInstance getDeclaringMacro() {
    // TODO: #19922 - We might replace Package#getDeclaringMacroForTarget by storing a reference to
    // the declaring macro in implementations of this interface (sharing memory with the field for
    // the package).
    return getPackage().getDeclaringMacroForTarget(getName());
  }

  /**
   * Returns the package that is considered to be the declaring location of this target.
   *
   * <p>For targets created inside a symbolic macro, this is the package containing the .bzl code of
   * the innermost running symbolic macro. For targets not in any symbolic macro, this is the same
   * as the package the target lives in.
   */
  default PackageIdentifier getDeclaringPackage() {
    PackageIdentifier pkgId = getPackage().getDeclaringPackageForTargetIfInMacro(getName());
    return pkgId != null ? pkgId : getPackage().getPackageIdentifier();
  }

  /**
   * Returns true if this target was declared within one or more symbolic macros, or false if it was
   * the product of running only a BUILD file and the legacy macros it called.
   */
  default boolean isCreatedInSymbolicMacro() {
    return getPackage().getDeclaringPackageForTargetIfInMacro(getName()) != null;
  }

  /**
   * Returns the rule associated with this target, if any.
   *
   * <p>If this is a Rule, returns itself; it this is an OutputFile, returns its generating rule; if
   * this is an input file, returns null.
   */
  @Nullable
  Rule getAssociatedRule();

  /**
   * Returns the license associated with this target.
   */
  License getLicense();

  /** Returns the set of distribution types associated with this target. */
  Set<DistributionType> getDistributions();

  /** Returns the visibility of this target. */
  // TODO(jhorvitz): Usually one of the following two methods suffice. Try to remove this.
  RuleVisibility getVisibility();

  /**
   * Equivalent to calling {@link RuleVisibility#getDependencyLabels} on the value returned by
   * {@link #getVisibility}, but potentially more efficient.
   *
   * <p>Prefer this method over {@link #getVisibility} when only the dependency labels are needed
   * and not a {@link RuleVisibility} instance.
   */
  default Iterable<Label> getVisibilityDependencyLabels() {
    return getVisibility().getDependencyLabels();
  }

  /**
   * Equivalent to calling {@link RuleVisibility#getDeclaredLabels} on the value returned by {@link
   * #getVisibility}, but potentially more efficient.
   *
   * <p>Prefer this method over {@link #getVisibility} when only the declared labels are needed and
   * not a {@link RuleVisibility} instance.
   */
  default List<Label> getVisibilityDeclaredLabels() {
    return getVisibility().getDeclaredLabels();
  }

  /** Returns whether this target type can be configured (e.g. accepts non-null configurations). */
  boolean isConfigurable();

  /**
   * Creates a compact representation of this target with enough information for dependent parents.
   */
  TargetData reduceForSerialization();
}
