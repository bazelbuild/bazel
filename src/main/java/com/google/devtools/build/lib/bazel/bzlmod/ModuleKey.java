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
import com.google.devtools.build.lib.cmdline.RepositoryName;

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
   * <p>NOTE(wyv): We don't prepend an '@' to the repo names of well-known modules. This is because
   * we still need the repo name to be 'bazel_tools' (not '@bazel_tools') since the command line
   * flags still don't go through repo mapping yet, and they're asking for '@bazel_tools//:thing',
   * not '@@bazel_tools//:thing'. We can't switch to the latter syntax because it doesn't work if
   * Bzlmod is not enabled. On the other hand, this means we cannot write '@@bazel_tools//:thing' to
   * bypass repo mapping at all, which can be awkward.
   *
   * <p>TODO(wyv): After we get rid of usage of com_google_protobuf in flag defaults, and make all
   * flag values go through repo mapping, we can remove the concept of well-known modules
   * altogether.
   */
  private static final ImmutableMap<String, RepositoryName> WELL_KNOWN_MODULES =
      ImmutableMap.of(
          "com_google_protobuf", RepositoryName.createUnvalidated("com_google_protobuf"),
          "protobuf", RepositoryName.createUnvalidated("com_google_protobuf"),
          "bazel_tools", RepositoryName.BAZEL_TOOLS,
          "local_config_platform", RepositoryName.createUnvalidated("local_config_platform"));

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
  public RepositoryName getCanonicalRepoName() {
    if (WELL_KNOWN_MODULES.containsKey(getName())) {
      return WELL_KNOWN_MODULES.get(getName());
    }
    if (ROOT.equals(this)) {
      return RepositoryName.MAIN;
    }
    return RepositoryName.createUnvalidated(
        String.format("@%s.%s", getName(), getVersion().isEmpty() ? "override" : getVersion()));
  }
}
