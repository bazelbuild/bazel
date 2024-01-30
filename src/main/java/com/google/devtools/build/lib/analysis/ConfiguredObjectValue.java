// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.skyframe.NotComparableSkyValue;

/**
 * Super-interface for {@link ConfiguredTargetValue} and {@link RuleConfiguredObjectValue}
 * (transitively including {@link AspectValue}).
 */
public interface ConfiguredObjectValue extends NotComparableSkyValue {
  /** Returns the configured target/aspect for this value. */
  ProviderCollection getConfiguredObject();

  /**
   * Returns the set of packages transitively loaded by this value. Must only be used for:
   *
   * <ul>
   *   <li>constructing the package -> source root map needed for some builds, OR
   *   <li>building the repo mapping manifest for runfiles
   * </ul>
   *
   * If the caller has not specified that this map needs to be constructed (via the constructor
   * argument in {@link
   * com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction#ConfiguredTargetFunction} or
   * {@link com.google.devtools.build.lib.skyframe.AspectFunction#AspectFunction}), calling this
   * will crash.
   */
  // TODO(b/283125139): Most builds never need to build a repo mapping manifest. Store transitive
  // packages outside of configured object values to save the wasted field.
  NestedSet<Package> getTransitivePackages();

  /**
   * Clears data from this value.
   *
   * <p>Should only be used when user specifies --discard_analysis_cache. Must be called at most
   * once per value, after which this object's other methods cannot be called.
   */
  void clear(boolean clearEverything);
}
