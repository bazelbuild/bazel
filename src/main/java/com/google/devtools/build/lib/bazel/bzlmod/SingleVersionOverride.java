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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;

/**
 * Specifies that the module should:
 *
 * <ul>
 *   <li>be pinned to a single version,
 *   <li>and/or come from a specific registry (instead of the default list),
 *   <li>and/or use some local patches.
 * </ul>
 */
@AutoValue
public abstract class SingleVersionOverride implements RegistryOverride {

  public static SingleVersionOverride create(
      Version version,
      String registry,
      ImmutableList<Label> patches,
      ImmutableList<String> patchCmds,
      int patchStrip) {
    return new AutoValue_SingleVersionOverride(version, registry, patches, patchCmds, patchStrip);
  }

  /**
   * The version to pin the module to. Can be empty if it shouldn't be pinned (in which case it will
   * still participate in version resolution).
   */
  public abstract Version getVersion();

  @Override
  public abstract String getRegistry();

  /** The labels of patches to apply after retrieving per the registry. */
  public abstract ImmutableList<Label> getPatches();

  /**
   * The patch commands to execute after retrieving per the registry. Should be a list of commands.
   */
  public abstract ImmutableList<String> getPatchCmds();

  /** The number of path segments to strip from the paths in the supplied patches. */
  public abstract int getPatchStrip();
}
