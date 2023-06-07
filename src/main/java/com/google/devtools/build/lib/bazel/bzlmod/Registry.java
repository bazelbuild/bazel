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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.io.IOException;
import java.util.Optional;

/** A database where module metadata is stored. */
public interface Registry {

  /** The URL that uniquely identifies the registry. */
  String getUrl();

  /**
   * Retrieves the contents of the module file of the module identified by {@code key} from the
   * registry. Returns {@code Optional.empty()} when the module is not found in this registry.
   */
  Optional<ModuleFile> getModuleFile(ModuleKey key, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException;

  /**
   * Retrieves the {@link RepoSpec} object that indicates how the contents of the module identified
   * by {@code key} should be materialized as a repo (with name {@code repoName}).
   */
  RepoSpec getRepoSpec(ModuleKey key, RepositoryName repoName, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException;

  /**
   * Retrieves yanked versions of the module identified by {@code key.getName()} from the registry.
   * Returns {@code Optional.empty()} when the information is not found in the registry.
   */
  Optional<ImmutableMap<Version, String>> getYankedVersions(
      String moduleName, ExtendedEventHandler eventHandler)
      throws IOException, InterruptedException;
}
