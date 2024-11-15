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

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule.ResolutionReason;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * Specifies that a module should be retrieved from a local directory.
 *
 * @param path The path to the local directory where the module contents should be found.
 */
@AutoCodec
public record LocalPathOverride(String path) implements NonRegistryOverride {
  public LocalPathOverride {
    requireNonNull(path, "path");
  }

  public static LocalPathOverride create(String path) {
    return new LocalPathOverride(path);
  }

  /** Returns the {@link RepoSpec} that defines this repository. */
  @Override
  public RepoSpec getRepoSpec() {
    return RepoSpec.builder()
        .setRuleClassName("local_repository")
        .setAttributes(AttributeValues.create(ImmutableMap.of("path", path())))
        .build();
  }

  @Override
  public ResolutionReason getResolutionReason() {
    return ResolutionReason.LOCAL_PATH_OVERRIDE;
  }
}
