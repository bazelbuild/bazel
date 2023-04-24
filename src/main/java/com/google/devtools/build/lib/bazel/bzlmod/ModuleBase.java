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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.RepositoryName;

/** Represents a node in the external dependency graph. */
abstract class ModuleBase {

  /**
   * The name of the module, as specified in this module's MODULE.bazel file. Can be empty if this
   * is the root module.
   */
  public abstract String getName();

  /**
   * The version of the module, as specified in this module's MODULE.bazel file. Can be empty if
   * this is the root module, or if this module comes from a {@link NonRegistryOverride}.
   */
  public abstract Version getVersion();

  /**
   * The key of this module in the dependency graph. Note that, although a {@link ModuleKey} is also
   * just a (name, version) pair, its semantics differ from {@link #getName} and {@link
   * #getVersion}, which are always as specified in the MODULE.bazel file. The {@link ModuleKey}
   * returned by this method, however, will have the following special semantics:
   *
   * <ul>
   *   <li>The name of the {@link ModuleKey} is the same as {@link #getName}, unless this is the
   *       root module, in which case the name of the {@link ModuleKey} must be empty.
   *   <li>The version of the {@link ModuleKey} is the same as {@link #getVersion}, unless this is
   *       the root module OR this module has a {@link NonRegistryOverride}, in which case the
   *       version of the {@link ModuleKey} must be empty.
   * </ul>
   */
  public abstract ModuleKey getKey();

  public final RepositoryName getCanonicalRepoName() {
    return getKey().getCanonicalRepoName();
  }

  /**
   * The name of the repository representing this module, as seen by the module itself. By default,
   * the name of the repo is the name of the module. This can be specified to ease migration for
   * projects that have been using a repo name for itself that differs from its module name.
   */
  public abstract String getRepoName();

  /**
   * Target patterns identifying execution platforms to register when this module is selected. Note
   * that these are what was written in module files verbatim, and don't contain canonical repo
   * names.
   */
  public abstract ImmutableList<String> getExecutionPlatformsToRegister();

  /**
   * Target patterns identifying toolchains to register when this module is selected. Note that
   * these are what was written in module files verbatim, and don't contain canonical repo names.
   */
  public abstract ImmutableList<String> getToolchainsToRegister();

  /** The module extensions used in this module. */
  public abstract ImmutableList<ModuleExtensionUsage> getExtensionUsages();
}
