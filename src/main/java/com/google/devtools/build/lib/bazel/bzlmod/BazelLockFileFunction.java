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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.gson.JsonSyntaxException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** Reads the contents of the lock file into its value. */
public class BazelLockFileFunction implements SkyFunction {

  public static final Precomputed<LockfileMode> LOCKFILE_MODE = new Precomputed<>("lockfile_mode");

  private static final Pattern LOCKFILE_VERSION_PATTERN =
      Pattern.compile("\"lockFileVersion\":\\s*(\\d+)");

  private final Path rootDirectory;

  private static final BzlmodFlagsAndEnvVars EMPTY_FLAGS =
      BzlmodFlagsAndEnvVars.create(
          ImmutableList.of(), ImmutableMap.of(), ImmutableList.of(), "", false, "", "");

  private static final BazelLockFileValue EMPTY_LOCKFILE =
      BazelLockFileValue.builder()
          .setLockFileVersion(BazelLockFileValue.LOCK_FILE_VERSION)
          .setModuleFileHash("")
          .setFlags(EMPTY_FLAGS)
          .setLocalOverrideHashes(ImmutableMap.of())
          .setModuleDepGraph(ImmutableMap.of())
          .setModuleExtensions(ImmutableMap.of())
          .build();

  public BazelLockFileFunction(Path rootDirectory) {
    this.rootDirectory = rootDirectory;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws BazelLockfileFunctionException, InterruptedException {
    RootedPath lockfilePath =
        RootedPath.toRootedPath(Root.fromPath(rootDirectory), LabelConstants.MODULE_LOCKFILE_NAME);

    // Add dependency on the lockfile to recognize changes to it
    if (env.getValue(FileValue.key(lockfilePath)) == null) {
      return null;
    }

    try {
      return getLockfileValue(lockfilePath);
    } catch (IOException | JsonSyntaxException | NullPointerException e) {
      throw new BazelLockfileFunctionException(
          ExternalDepsException.withMessage(
              Code.BAD_MODULE,
              "Failed to read and parse the MODULE.bazel.lock file with error: %s."
                  + " Try deleting it and rerun the build.",
              e.getMessage()),
          Transience.PERSISTENT);
    }
  }

  public static BazelLockFileValue getLockfileValue(RootedPath lockfilePath) throws IOException {
    BazelLockFileValue bazelLockFileValue;
    try {
      String json = FileSystemUtils.readContent(lockfilePath.asPath(), UTF_8);
      Matcher matcher = LOCKFILE_VERSION_PATTERN.matcher(json);
      int version = matcher.find() ? Integer.parseInt(matcher.group(1)) : -1;
      if (version == BazelLockFileValue.LOCK_FILE_VERSION) {
        bazelLockFileValue =
            GsonTypeAdapterUtil.createLockFileGson(
                    lockfilePath
                        .asPath()
                        .getParentDirectory()
                        .getRelative(LabelConstants.MODULE_DOT_BAZEL_FILE_NAME))
                .fromJson(json, BazelLockFileValue.class);
      } else {
        // This is an old version, needs to be updated
        // Keep old version to recognize the problem in error mode
        bazelLockFileValue = EMPTY_LOCKFILE.toBuilder().setLockFileVersion(version).build();
      }
    } catch (FileNotFoundException e) {
      bazelLockFileValue = EMPTY_LOCKFILE;
    }
    return bazelLockFileValue;
  }


  static final class BazelLockfileFunctionException extends SkyFunctionException {

    BazelLockfileFunctionException(ExternalDepsException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
