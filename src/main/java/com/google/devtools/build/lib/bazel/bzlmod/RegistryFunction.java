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
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.DirectoryTreeDigestValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.time.Duration;
import java.time.Instant;
import java.util.Optional;
import javax.annotation.Nullable;

/** A simple SkyFunction that creates a {@link Registry} with a given URL. */
public class RegistryFunction implements SkyFunction {
  /**
   * Set to the current time in {@link com.google.devtools.build.lib.bazel.BazelRepositoryModule}
   * after {@link #INVALIDATION_INTERVAL} has passed. This is used to refresh the mutable registry
   * contents cached in memory from time to time.
   */
  public static final Precomputed<Instant> LAST_INVALIDATION =
      new Precomputed<>("last_registry_invalidation");

  public static final Precomputed<ImmutableMap<String, ImmutableSet<String>>> MODULE_MIRRORS =
      new Precomputed<>("module_mirrors");

  /**
   * URLs of the local registries passed with {@code --watched_registry}, with {@code %workspace%}
   * already expanded.
   */
  public static final Precomputed<ImmutableSet<String>> WATCHED_REGISTRIES =
      new Precomputed<>("watched_registries");

  /**
   * The interval after which the mutable registry contents cached in memory should be refreshed.
   */
  public static final Duration INVALIDATION_INTERVAL = Duration.ofHours(1);

  private final RegistryFactory registryFactory;
  private final Path workspaceRoot;

  public RegistryFunction(RegistryFactory registryFactory, Path workspaceRoot) {
    this.registryFactory = registryFactory;
    this.workspaceRoot = workspaceRoot;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, RegistryException {
    LockfileMode lockfileMode = BazelLockFileFunction.LOCKFILE_MODE.get(env);
    Optional<Path> vendorDir = RepositoryDirectoryValue.VENDOR_DIRECTORY.get(env);

    if (lockfileMode == LockfileMode.REFRESH) {
      LAST_INVALIDATION.get(env);
    }

    BazelLockFileValue lockfile = (BazelLockFileValue) env.getValue(BazelLockFileValue.KEY);
    if (lockfile == null) {
      return null;
    }

    RegistryKey key = (RegistryKey) skyKey.argument();
    String url = key.url().replace("%workspace%", workspaceRoot.getPathString());
    try {
      if (WATCHED_REGISTRIES.get(env).contains(url) && !addLocalRegistryTreeDependency(url, env)) {
        return null;
      }
      return registryFactory.createRegistry(
          url,
          lockfileMode,
          lockfile.getRegistryFileHashes(),
          lockfile.getSelectedYankedVersions(),
          vendorDir,
          MODULE_MIRRORS.get(env).getOrDefault(key.url(), ImmutableSet.of()));
    } catch (URISyntaxException e) {
      throw new RegistryException(
          ExternalDepsException.withCauseAndMessage(
              FailureDetails.ExternalDeps.Code.INVALID_REGISTRY_URL,
              e,
              "Invalid registry URL: %s",
              key.url()));
    }
  }

  /**
   * Local registry files are read outside of Skyframe, so nothing would otherwise invalidate the
   * {@link Registry} (and the values computed from its files) when they change. For watched
   * registries, this requests the digest of the registry tree purely to add a Skyframe dependency
   * on it; the value itself is unused. Other local registries are not watched, since every file in
   * the tree is then checked for changes on each command.
   *
   * @return false if the digest has not been computed yet and the caller must return null
   */
  private boolean addLocalRegistryTreeDependency(String url, Environment env)
      throws URISyntaxException, InterruptedException, RegistryException {
    RootedPath registryRoot =
        RootedPath.toRootedPath(
            Root.absoluteRoot(workspaceRoot.getFileSystem()), getWatchedRegistryPath(url));
    try {
      return env.getValueOrThrow(
              DirectoryTreeDigestValue.key(registryRoot, registryRoot, ImmutableList.of()),
              IOException.class)
          != null;
    } catch (IOException e) {
      throw new RegistryException(
          ExternalDepsException.withCauseAndMessage(
              FailureDetails.ExternalDeps.Code.ERROR_ACCESSING_REGISTRY,
              e,
              "Failed to read local registry %s",
              url));
    }
  }

  /**
   * Returns the absolute path of the directory of a watched registry, given its URL with {@code
   * %workspace%} expanded.
   *
   * @throws URISyntaxException if the URL is not a {@code file://} URL with an absolute path
   */
  public static PathFragment getWatchedRegistryPath(String url) throws URISyntaxException {
    URI uri = new URI(url);
    if (!"file".equals(uri.getScheme())) {
      throw new URISyntaxException(url, "Only file:// registries can be watched");
    }
    PathFragment path = PathFragment.create(IndexRegistry.getLocalRegistryPath(uri));
    if (!path.isAbsolute()) {
      // For example file://C:/foo on Windows, which should be file:///C:/foo.
      throw new URISyntaxException(
          url,
          "Local registry URL must have an absolute path -- did you mean to use file:///foo/bar"
              + " or file:///c:/foo/bar for Windows?");
    }
    return path;
  }

  static final class RegistryException extends SkyFunctionException {

    RegistryException(ExternalDepsException cause) {
      super(cause, Transience.TRANSIENT);
    }
  }
}
