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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;

/**
 * Provider class for configured targets that have a visibility.
 *
 * <p>The visibility provider is computed in {@link ConfiguredTargetFactory#convertVisibility}.
 * Essentially, it starts with the target's {@link RuleVisibility} from the loading phase, resolves
 * the references to {@code package_group}s (which may be aggregated through multiple {@code
 * includes}), and merges all the visibility grants into a {@code NestedSet} of {@link
 * PackageGroupContents}. Thus, the visibility provider contains transitive information, which gets
 * flattened at the time of visibility checking.
 *
 * <p>This provider also needs to track a bit indicating whether the target was declared within one
 * or more symbolic macros. This is used by {@link CommonPrerequisiteValidator} to help implement
 * the visibility check in two ways:
 *
 * <ul>
 *   <li>First, for targets not in symbolic macros, the declaration location (i.e. package) of the
 *       target is not concatenated into its visibility attribute (see {@link
 *       Rule#getRuleVisibility}), and {@code CommonPrerequisiteValidator} needs to know to account
 *       for this when doing the visibility check.
 *   <li>Second, there's the {@link CommonPrerequisiteValidator#isSameLogicalPackage} hook, which
 *       powers the hack that targets in {@code //javatests/foo} are allowed to see targets in
 *       {@code //java/foo}. (This feature is only active within Google and disabled for OSS Bazel.)
 *       The semantics are that targets created in symbolic macros are never automatically visible
 *       to {@code //javatests/foo} packages, regardless of the package or declaration location.
 * </ul>
 */
public interface VisibilityProvider extends TransitiveInfoProvider {

  NestedSet<PackageGroupContents> PUBLIC_VISIBILITY =
      NestedSetBuilder.create(
          Order.STABLE_ORDER,
          PackageGroupContents.create(ImmutableList.of(PackageSpecification.everything())));

  NestedSet<PackageGroupContents> PRIVATE_VISIBILITY =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);

  /**
   * Returns the visibility specification, as determined by resolving the entries in the {@code
   * visibility} attribute.
   *
   * <p>For targets in symbolic macros, this will include the target's declaration location. For
   * targets created directly by the package (including in legacy macros that are not in symbolic
   * macros), it may not include the target's package, even though it is technically visible to it.
   */
  NestedSet<PackageGroupContents> getVisibility();

  /**
   * Returns whether this target was instantiated in one or more symbolic macros.
   *
   * <p>(This information can be determined from the {@link Rule} object, but that's not necessarily
   * available from a prerequisite object during analysis.)
   */
  boolean isCreatedInSymbolicMacro();
}
