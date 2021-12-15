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
import com.google.common.collect.ImmutableMap;

/** A module name, version pair that identifies a module in the external dependency graph. */
@AutoValue
public abstract class ModuleKey {

  /**
   * A mapping from module name to repository name.
   *
   * <p>For some well known modules, their repository names are referenced in default label values
   * of some native rules' attributes and command line flags, which don't go through repo mappings.
   * Therefore, we have to keep its canonical repository name the same as its well known repository
   * name. Eg. "@com_google_protobuf//:protoc" is used for --proto_compiler flag.
   *
   * <p>TODO(pcloudy): Remove this hack after figuring out a correct way to deal with the above
   * situation.
   */
  private static final ImmutableMap<String, String> WELL_KNOWN_MODULES =
      ImmutableMap.of(
          "com_google_protobuf", "com_google_protobuf", "protobuf", "com_google_protobuf");

  public static final ModuleKey ROOT = create("", Version.EMPTY);

  public static ModuleKey create(String name, Version version) {
    return new AutoValue_ModuleKey(name, version);
  }

  /** The name of the module. Must be empty for the root module. */
  public abstract String getName();

  /** The version of the module. Must be empty iff the module has a {@link NonRegistryOverride}. */
  public abstract Version getVersion();

  @Override
  public final String toString() {
    if (this.equals(ROOT)) {
      return "<root>";
    }
    return getName() + "@" + (getVersion().isEmpty() ? "_" : getVersion().toString());
  }

  /** Returns the canonical name of the repo backing this module. */
  public String getCanonicalRepoName() {
    if (WELL_KNOWN_MODULES.containsKey(getName())) {
      return WELL_KNOWN_MODULES.get(getName());
    }
    if (getVersion().isEmpty()) {
      return getName();
    }
    return getName() + "." + getVersion();
  }
}
