// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;

/**
 * A module name, version, and max compatibility level tuple that represents a desired module in the
 * external dependency graph.
 */
@AutoValue
public abstract class UnresolvedModuleKey {

  public static final UnresolvedModuleKey ROOT = create("", Version.EMPTY, 0);

  public static UnresolvedModuleKey create(
      String name, Version version, int maxCompatibilityLevel) {
    return new AutoValue_UnresolvedModuleKey(name, version, maxCompatibilityLevel);
  }

  /** The name of the module. Must be empty for the root module. */
  public abstract String getName();

  /** The version of the module. Must be empty iff the module has a {@link NonRegistryOverride}. */
  public abstract Version getVersion();

  /** The maximum compatibility level of the module. */
  public abstract int getMaxCompatibilityLevel();

  @Override
  public final String toString() {
    if (this.equals(ROOT)) {
      return "<root>";
    }
    return getName()
        + "@"
        + (getVersion().isEmpty() ? "_" : getVersion().toString())
        + (getMaxCompatibilityLevel() > 0
            ? "[max_compatibility_level=" + getMaxCompatibilityLevel() + "]"
            : "");
  }

  public ModuleKey getMinCompatibilityModuleKey() {
    return ModuleKey.create(getName(), getVersion());
  }
}
