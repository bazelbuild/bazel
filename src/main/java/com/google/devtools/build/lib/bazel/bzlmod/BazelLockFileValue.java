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


import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.util.Arrays;
import java.util.Map;

/**
 * The result of reading the lockfile. Contains the lockfile version, module hash, definitions of
 * module repositories, post-resolution dependency graph and module extensions data (ID, hash,
 * definition, usages)
 */
@AutoValue
@GenerateTypeAdapter
public abstract class BazelLockFileValue implements SkyValue, Postable {

  public static final int LOCK_FILE_VERSION = 1;

  @SerializationConstant public static final SkyKey KEY = () -> SkyFunctions.BAZEL_LOCK_FILE;

  static Builder builder() {
    return new AutoValue_BazelLockFileValue.Builder()
        .setLockFileVersion(LOCK_FILE_VERSION)
        .setModuleExtensions(ImmutableMap.of());
  }

  /** Current version of the lock file */
  public abstract int getLockFileVersion();

  /** Hash of the Module file */
  public abstract String getModuleFileHash();

  /** Command line flags and environment variables that can affect the resolution */
  public abstract BzlmodFlagsAndEnvVars getFlags();

  /** Module hash of each local path override in the root module file */
  public abstract ImmutableMap<String, String> getLocalOverrideHashes();

  /** The post-selection dep graph retrieved from the lock file. */
  public abstract ImmutableMap<ModuleKey, Module> getModuleDepGraph();

  /** Mapping the extension id to the module extension data */
  public abstract ImmutableMap<ModuleExtensionId, LockFileModuleExtension> getModuleExtensions();

  public abstract Builder toBuilder();

  /** Builder type for {@link BazelLockFileValue}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setLockFileVersion(int value);

    public abstract Builder setModuleFileHash(String value);

    public abstract Builder setFlags(BzlmodFlagsAndEnvVars value);

    public abstract Builder setLocalOverrideHashes(ImmutableMap<String, String> value);

    public abstract Builder setModuleDepGraph(ImmutableMap<ModuleKey, Module> value);

    public abstract Builder setModuleExtensions(
        ImmutableMap<ModuleExtensionId, LockFileModuleExtension> value);

    public abstract BazelLockFileValue build();
  }

  /** Returns the difference between the lockfile and the current module & flags */
  public ImmutableList<String> getModuleAndFlagsDiff(
      String moduleFileHash,
      ImmutableMap<String, String> localOverrideHashes,
      BzlmodFlagsAndEnvVars flags) {
    ImmutableList.Builder<String> moduleDiff = new ImmutableList.Builder<>();
    if (getLockFileVersion() != BazelLockFileValue.LOCK_FILE_VERSION) {
      return moduleDiff
          .add(
              "the version of the lockfile is not compatible with the current Bazel, please run"
                  + " with '--lockfile_mode=update'")
          .build();
    }
    if (!moduleFileHash.equals(getModuleFileHash())) {
      moduleDiff.add("the root MODULE.bazel has been modified");
    }
    moduleDiff.addAll(getFlags().getDiffFlags(flags));

    for (Map.Entry<String, String> entry : localOverrideHashes.entrySet()) {
      String currentValue = entry.getValue();
      String lockfileValue = getLocalOverrideHashes().get(entry.getKey());
      // If the lockfile value is null, the module hash would be different anyway
      if (lockfileValue != null && !currentValue.equals(lockfileValue)) {
        moduleDiff.add(
            "The MODULE.bazel file has changed for the overriden module: " + entry.getKey());
      }
    }
    return moduleDiff.build();
  }

  /** Returns the differences between an extension and its locked data */
  public ImmutableList<String> getModuleExtensionDiff(
      ModuleExtensionId extensionId,
      byte[] transitiveDigest,
      boolean filesChanged,
      ImmutableMap<String, String> envVariables,
      ImmutableMap<ModuleKey, ModuleExtensionUsage> extensionUsages,
      ImmutableMap<ModuleKey, ModuleExtensionUsage> lockedExtensionUsages) {
    LockFileModuleExtension lockedExtension = getModuleExtensions().get(extensionId);

    ImmutableList.Builder<String> extDiff = new ImmutableList.Builder<>();
    if (!Arrays.equals(transitiveDigest, lockedExtension.getBzlTransitiveDigest())) {
        extDiff.add(
            "The implementation of the extension '"
                + extensionId
                + "' or one of its transitive .bzl files has changed");
    }
    if (filesChanged) {
      extDiff.add("One or more files the extension '" + extensionId + "' is using have changed");
    }
    if (!extensionUsages.equals(lockedExtensionUsages)) {
      extDiff.add("The usages of the extension '" + extensionId + "' has changed");
    }
    if (!envVariables.equals(lockedExtension.getEnvVariables())) {
      extDiff.add(
          "The environment variables the extension '"
              + extensionId
              + "' depends on (or their values) have changed");
    }
    return extDiff.build();
  }
}
