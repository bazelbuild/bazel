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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.util.Map;

/**
 * Represents a use of a symbolic macro in a package.
 *
 * <p>There is one {@code MacroInstance} for each call to a {@link
 * StarlarkRuleClassFunctions#MacroFunction} that is executed during a package's evaluation. Just as
 * a {@link MacroClass} is analogous to a {@link RuleClass}, {@code MacroInstance} is analogous to a
 * {@link Rule} (i.e. a rule target).
 */
public final class MacroInstance {

  private final MacroClass macroClass;
  // TODO(#19922): Consider switching to more optimized, indexed representation, as in Rule.
  // Order isn't guaranteed, sort before dumping.
  private final ImmutableMap<String, Object> attrValues;

  public MacroInstance(MacroClass macroClass, Map<String, Object> attrValues) {
    this.macroClass = macroClass;
    this.attrValues = ImmutableMap.copyOf(attrValues);
    Preconditions.checkArgument(macroClass.getAttributes().keySet().equals(attrValues.keySet()));
  }

  /** Returns the {@link MacroClass} (i.e. schema info) that this instance parameterizes. */
  public MacroClass getMacroClass() {
    return macroClass;
  }

  /**
   * Returns the name of this instance, as given in the {@code name = ...} attribute in the calling
   * BUILD file or macro.
   */
  public String getName() {
    // Type enforced by RuleClass.NAME_ATTRIBUTE.
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
}
