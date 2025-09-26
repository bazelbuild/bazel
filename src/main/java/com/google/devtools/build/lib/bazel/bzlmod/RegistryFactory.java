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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.vfs.Path;
import java.net.URISyntaxException;
import java.util.Optional;

/** A factory type for {@link Registry}. */
public interface RegistryFactory {

  /**
   * Creates a registry associated with the given URL.
   *
   * <p>Outside of tests, only {@link RegistryFunction} should call this method.
   */
  Registry createRegistry(
      String url,
      RepositoryOptions.LockfileMode lockfileMode,
      ImmutableMap<String, Optional<Checksum>> fileHashes,
      ImmutableMap<ModuleKey, String> previouslySelectedYankedVersions,
      Optional<Path> vendorDir,
      ImmutableSet<String> moduleMirrors)
      throws URISyntaxException;
}
