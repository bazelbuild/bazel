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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.devtools.build.lib.bazel.bzlmod.GsonTypeAdapterUtil.LOCKFILE_GSON;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Module collecting Bazel module and module extensions resolution results and updating the
 * lockfile.
 */
public class BazelLockFileModule extends BlazeModule {

  private Path workspaceRoot;

  @Nullable private BazelLockFileValue lockfileValue;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  @Override
  public void beforeCommand(CommandEnvironment env) {
    workspaceRoot = env.getWorkspace();
    RepositoryOptions options = env.getOptions().getOptions(RepositoryOptions.class);
    if (options.lockfileMode.equals(LockfileMode.UPDATE)) {
      env.getEventBus().register(this);
    }
  }

  @Override
  public void afterCommand() {
    if (lockfileValue == null) { // module didn't change --> no event was posted
      return;
    }
    updateLockfile(workspaceRoot, lockfileValue);
    this.lockfileValue = null;
  }

  /**
   * Updates the data stored in the lockfile (MODULE.bazel.lock)
   *
   * @param workspaceRoot Where to update the lockfile
   * @param updatedLockfile The updated lockfile data to save
   */
  public static void updateLockfile(Path workspaceRoot, BazelLockFileValue updatedLockfile) {
    RootedPath lockfilePath =
        RootedPath.toRootedPath(Root.fromPath(workspaceRoot), LabelConstants.MODULE_LOCKFILE_NAME);
    try {
      FileSystemUtils.writeContent(
          lockfilePath.asPath(), UTF_8, LOCKFILE_GSON.toJson(updatedLockfile));
    } catch (IOException e) {
      logger.atSevere().withCause(e).log(
          "Error while updating MODULE.bazel.lock file: %s", e.getMessage());
    }
  }

  @Subscribe
  public void bazelModuleResolved(BazelLockFileValue lockfileValue) {
    this.lockfileValue = lockfileValue;
  }
}
