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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.NotComparableSkyValue;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/** A database where module metadata is stored. */
public interface Registry extends NotComparableSkyValue {

  /** The URL that uniquely identifies the registry. */
  String getUrl();

  /** Thrown when a file is not found in the registry. */
  final class NotFoundException extends Exception {
    public NotFoundException(String message) {
      super(message);
    }
  }

  /**
   * Retrieves the contents of the module file of the module identified by {@code key} from the
   * registry.
   *
   * @throws NotFoundException if the module file is not found in the registry
   */
  ModuleFile getModuleFile(
      ModuleKey key, ExtendedEventHandler eventHandler, DownloadManager downloadManager)
      throws IOException, InterruptedException, NotFoundException;

  /**
   * Retrieves the {@link RepoSpec} object that indicates how the contents of the module identified
   * by {@code key} should be materialized as a repo.
   */
  RepoSpec getRepoSpec(
      ModuleKey key,
      ImmutableMap<String, Optional<Checksum>> moduleFileHashes,
      ExtendedEventHandler eventHandler,
      DownloadManager downloadManager)
      throws IOException, InterruptedException;

  /**
   * Retrieves yanked versions of the module identified by {@code key.getName()} from the registry.
   * Returns {@code Optional.empty()} when the information is not found in the registry.
   */
  Optional<ImmutableMap<Version, String>> getYankedVersions(
      String moduleName, ExtendedEventHandler eventHandler, DownloadManager downloadManager)
      throws IOException, InterruptedException;

  /**
   * Returns the yanked versions information, limited to the given selected module version, purely
   * based on the lockfile (if possible).
   */
  Optional<YankedVersionsValue> tryGetYankedVersionsFromLockfile(ModuleKey selectedModuleKey);

  /**
   * The versions of a module listed by a single registry: {@code available} holds the non-yanked
   * versions, {@code yanked} the yanked ones.
   */
  record KnownVersions(ImmutableList<Version> available, ImmutableSet<Version> yanked) {}

  /**
   * Retrieves the versions of the module identified by {@code moduleName} that this registry lists.
   * Returns {@code Optional.empty()} when the registry has no information about the module.
   */
  Optional<KnownVersions> getKnownVersions(
      String moduleName, ExtendedEventHandler eventHandler, DownloadManager downloadManager)
      throws IOException, InterruptedException;

  /**
   * Merges per-registry version information, given in registry precedence order, into the list of
   * versions an upgrade may target. Module resolution fetches a version from the first registry
   * that has it, so a version counts as available only if the first registry listing it (whether as
   * available or as yanked) lists it as available. Returns {@code Optional.empty()} when no
   * registry lists the module.
   */
  static Optional<ImmutableList<Version>> mergeKnownVersions(
      List<Optional<KnownVersions>> perRegistry) {
    boolean listed = false;
    Set<Version> seen = new HashSet<>();
    ImmutableList.Builder<Version> available = ImmutableList.builder();
    for (Optional<KnownVersions> knownVersions : perRegistry) {
      if (knownVersions.isEmpty()) {
        continue;
      }
      listed = true;
      for (Version version : knownVersions.get().available()) {
        if (seen.add(version)) {
          available.add(version);
        }
      }
      seen.addAll(knownVersions.get().yanked());
    }
    return listed ? Optional.of(available.build()) : Optional.empty();
  }
}
