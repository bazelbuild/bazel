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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.starlarkbuildapi.TargetApi;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/**
 * A node in the build dependency graph, identified by a Label.
 *
 * <p>This StarlarkBuiltin does not contain any documentation since Starlark's Target type refers to
 * TransitiveInfoCollection.class, which contains the appropriate documentation.
 */
public interface Target extends TargetApi {

  /**
   *  Returns the Package to which this rule belongs.
   */
  Package getPackage();

  /**
   * Returns a string describing this kind of target: e.g. "cc_library rule",
   * "source file", "generated file".
   */
  String getTargetKind();

  /**
   * Returns the rule associated with this target, if any.
   *
   * If this is a Rule, returns itself; it this is an OutputFile, returns its
   * generating rule; if this is an input file, returns null.
   */
  @Nullable
  Rule getAssociatedRule();

  /**
   * Returns the license associated with this target.
   */
  License getLicense();

  /**
   * Returns the place where the target was defined.
   *
   * <p>The location of a rule instance is generally its "generator location", the location of the
   * outermost call on the stack, which is in the BUILD file. However, the location of a source file
   * target created by an explicit call to {@code exports_files} is the location of the innermost
   * call, which may be in an arbitrary .bzl file and is not necessarily beneath the package's
   * directory. The inconsistency seems unintentional.
   */
  Location getLocation();

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

  /** Returns the rule class name if the target is a rule and {@code ""} otherwise. */
  default String getRuleClass() {
    return "";
  }

  /** Returns the rule tags if the target is a rule and an empty set otherwise. */
  default Set<String> getRuleTags() {
    return ImmutableSet.of();
  }
}
