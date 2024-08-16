// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import java.util.Map;

/**
 * Represents a use of a symbolic macro in a package.
 *
 * <p>There is one {@code MacroInstance} for each call to a {@link
 * StarlarkRuleClassFunctions#MacroFunction} that is executed during a package's evaluation. Just as
 * a {@link MacroClass} is analogous to a {@link RuleClass}, {@code MacroInstance} is analogous to a
 * {@link Rule} (i.e. a rule target).
 *
 * <p>Macro instance names are not guaranteed to be unique within a package; see {@link #getId}.
 */
public final class MacroInstance {

  private final Package pkg;

  private final MacroClass macroClass;

  private final int sameNameDepth;

  // TODO(#19922): Consider switching to more optimized, indexed representation, as in Rule.
  // Order isn't guaranteed, sort before dumping.
  private final ImmutableMap<String, Object> attrValues;

  /**
   * Instantiates the given macro class with the given attribute values.
   *
   * <p>{@code sameNameDepth} is the number of macro instances that this one is inside of that share
   * its name. For most instances it is 1, but for the main submacro of a parent macro it is one
   * more than the parent's depth.
   */
  public MacroInstance(
      Package pkg, MacroClass macroClass, Map<String, Object> attrValues, int sameNameDepth) {
    this.pkg = pkg;
    this.macroClass = macroClass;
    this.attrValues = ImmutableMap.copyOf(attrValues);
    Preconditions.checkArgument(sameNameDepth > 0);
    this.sameNameDepth = sameNameDepth;
    Preconditions.checkArgument(macroClass.getAttributes().keySet().equals(attrValues.keySet()));
  }

  /** Returns the package this instance was created in. */
  public Package getPackage() {
    return pkg;
  }

  /** Returns the {@link MacroClass} (i.e. schema info) that this instance parameterizes. */
  public MacroClass getMacroClass() {
    return macroClass;
  }

  /**
   * The depth of this macro instance in a chain of nested macros having the same name.
   *
   * <p>1 for any macro that is not declared in a macro of the same name.
   *
   * <p>Used by {@link #getId}.
   */
  public int getSameNameDepth() {
    return sameNameDepth;
  }

  /**
   * Returns the id of this macro instance. The id is the name, concatenated with {@code ":n"} where
   * n is an integer distinguishing this from other macro instances of the same name in the package.
   *
   * <p>Within a package, two macro instances are not allowed to share the same name except when one
   * of them is the main submacro of the other. More generally, there may be a contiguous chain of
   * nested main submacros that all share the same name, but these may not share with any other
   * macro outside the chain. We allow this exception so that the build does not break if the rule
   * of a main target is refactored into a macro. The tradeoff of this design is that the name alone
   * is not enough to disambiguate between macros in the chain.
   *
   * <p>The number n is simply the depth of the macro in the chain of same-named macros, starting at
   * 1. For example, if we have a chain of macro expansions foo -> foo_bar -> foo_bar -> foo_bar ->
   * foo_bar_baz, then the ids of these macros are respectively "foo:1", "foo_bar:1", "foo_bar:2",
   * "foo_bar:3", "foo_bar_baz:1".
   *
   * <p>Note that ids only serve to canonically identify macro instances, and play no role in naming
   * or name conflict detection.
   */
  public String getId() {
    return getName() + ":" + sameNameDepth;
  }

  /**
   * Returns the name of this instance, as given in the {@code name = ...} attribute in the calling
   * BUILD file or macro.
   */
  public String getName() {
    // Type and existence enforced by RuleClass.NAME_ATTRIBUTE.
    return (String) Preconditions.checkNotNull(attrValues.get("name"));
  }

  /**
   * Dictionary of attributes for this instance.
   *
   * <p>Contains all attributes, as seen after processing by {@link
   * MacroClass#instantiateAndAddMacro}.
   */
  public ImmutableMap<String, Object> getAttrValues() {
    return attrValues;
  }

  /**
   * Logical tuple of the package and id within the package. Used to label the Starlark evaluation
   * environment.
   */
  @AutoValue
  abstract static class UniqueId {
    static UniqueId create(PackageIdentifier packageId, String id) {
      return new AutoValue_MacroInstance_UniqueId(packageId, id);
    }

    abstract PackageIdentifier packageId();

    abstract String id();
  }
}
