// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;

/**
 * Helper that reads the bzlmod lockfile and answers: "Given a generated repo name, what files did
 * the extension read to create it?"
 */
public final class BzlmodExtensionInputsHelper {

  private final BazelLockFileValue lockfile;

  private BzlmodExtensionInputsHelper(BazelLockFileValue lockfile) {
    this.lockfile = lockfile;
  }

  /**
   * Creates a helper from a lockfile. Called by SkyQueryEnvironment after it loads the lockfile
   * from Skyframe.
   */
  public static BzlmodExtensionInputsHelper create(BazelLockFileValue lockfile) {
    return new BzlmodExtensionInputsHelper(lockfile);
  }

  /**
   * Given a generated repo name (e.g. "com_github_foo"), returns the file paths that the extension
   * recorded reading when it created that repo.
   */
  public ImmutableSet<RepoRecordedInput.RepoCacheFriendlyPath> getRecordedFilesForRepo(
      String repoName) {
    ImmutableSet.Builder<RepoRecordedInput.RepoCacheFriendlyPath> result =
        ImmutableSet.builder();

    for (var outerEntry : lockfile.getModuleExtensions().entrySet()) {
      for (LockFileModuleExtension ext : outerEntry.getValue().values()) {
        if (!ext.getGeneratedRepoSpecs().containsKey(repoName)) {
          continue;
        }
        for (RepoRecordedInput.WithValue wv : ext.getRecordedInputs()) {
          if (wv.input() instanceof RepoRecordedInput.File f) {
            result.add(f.getPath());
          }
        }
      }
    }

    return result.build();
  }
}
