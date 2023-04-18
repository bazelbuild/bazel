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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.util.ArrayList;

/**
 * The result of reading the lockfile. Contains the lockfile version, module hash, definitions of
 * module repositories, post-resolution dependency graph and module extensions data (ID, hash,
 * definition, usages)
 */
@AutoValue
@GenerateTypeAdapter
public abstract class BazelLockFileValue implements SkyValue {

  public static final int LOCK_FILE_VERSION = 1;

  @SerializationConstant public static final SkyKey KEY = () -> SkyFunctions.BAZEL_LOCK_FILE;

  public static BazelLockFileValue create(
      int lockFileVersion,
      String moduleFileHash,
      BzlmodFlagsAndEnvVars flags,
      ImmutableMap<ModuleKey, Module> moduleDepGraph) {
    return new AutoValue_BazelLockFileValue(lockFileVersion, moduleFileHash, flags, moduleDepGraph);
  }

  /** Current version of the lock file */
  public abstract int getLockFileVersion();

  /** Hash of the Module file */
  public abstract String getModuleFileHash();

  /** Command line flags and environment variables that can affect the resolution */
  public abstract BzlmodFlagsAndEnvVars getFlags();

  /** The post-selection dep graph retrieved from the lock file. */
  public abstract ImmutableMap<ModuleKey, Module> getModuleDepGraph();

  /** Returns the difference between the lockfile and the current module & flags */
  public ArrayList<String> getDiffLockfile(String moduleFileHash, BzlmodFlagsAndEnvVars flags) {
    ArrayList<String> diffLockfile = new ArrayList<>();
    if (!moduleFileHash.equals(getModuleFileHash())) {
      diffLockfile.add("the root MODULE.bazel has been modified");
    }
    if (!flags.cmdRegistries().equals(getFlags().cmdRegistries())) {
      diffLockfile.add("the value of --registry flag has been modified");
    }
    if (!flags.cmdModuleOverrides().equals(getFlags().cmdModuleOverrides())) {
      diffLockfile.add("the value of --override_module flag has been modified");
    }
    if (!flags.allowedYankedVersions().equals(getFlags().allowedYankedVersions())) {
      diffLockfile.add("the value of --allow_yanked_versions flag has been modified");
    }
    if (!flags.envVarAllowedYankedVersions().equals(getFlags().envVarAllowedYankedVersions())) {
      diffLockfile.add(
          "the value of BZLMOD_ALLOW_YANKED_VERSIONS environment variable has been modified");
    }
    if (flags.ignoreDevDependency() != getFlags().ignoreDevDependency()) {
      diffLockfile.add("the value of --ignore_dev_dependency flag has been modified");
    }
    if (!flags.directDependenciesMode().equals(getFlags().directDependenciesMode())) {
      diffLockfile.add("the value of --check_direct_dependencies flag has been modified");
    }
    if (!flags.compatibilityMode().equals(getFlags().compatibilityMode())) {
      diffLockfile.add("the value of --check_bazel_compatibility flag has been modified");
    }
    return diffLockfile;
  }
}
