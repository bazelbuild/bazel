// Copyright 2021 The Bazel Authors. All rights reserved.
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

/** A module name, version pair that identifies a module in the external dependency graph. */
@AutoValue
public abstract class ModuleKey {

  public static ModuleKey create(String name, Version version) {
    return new AutoValue_ModuleKey(name, version);
  }

  /** The name of the module. Can be empty for the root module (if it doesn't specify a name). */
  public abstract String getName();

  /** The version of the module. Must be empty iff the module has a {@link NonRegistryOverride}. */
  public abstract Version getVersion();

  @Override
  public final String toString() {
    return (getName().isEmpty() ? "_" : getName())
        + "@"
        + (getVersion().isEmpty() ? "_" : getVersion().toString());
  }

  /** Returns the canonical name of the repo backing this module. */
  public String getCanonicalRepoName() {
    return getName() + "." + getVersion();
  }
}
