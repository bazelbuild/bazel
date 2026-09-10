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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Optional;
import net.starlark.java.eval.Dict;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BzlmodExtensionInputsHelper}. */
@RunWith(JUnit4.class)
public class BzlmodExtensionInputsHelperTest {

  @Test
  public void getRecordedFilesForRepo_returnsFilesReadByExtension() throws Exception {
    RepoRecordedInput.RepoCacheFriendlyPath goModPath =
        RepoRecordedInput.RepoCacheFriendlyPath.createInsideWorkspace(
            RepositoryName.MAIN, PathFragment.create("go.mod"));

    RepoRecordedInput.WithValue goModInput =
        new RepoRecordedInput.WithValue(new RepoRecordedInput.File(goModPath), "abc123");

    LockFileModuleExtension fakeExtension =
        LockFileModuleExtension.builder()
            .setBzlTransitiveDigest(new byte[0])
            .setUsagesDigest(new byte[0])
            .setRecordedInputs(ImmutableList.of(goModInput))
            .setGeneratedRepoSpecs(
                ImmutableMap.of(
                    "com_github_foo",
                    new RepoSpec(
                        new RepoRuleId(
                            Label.parseCanonicalUnchecked("//:ext.bzl"), "go_repository"),
                        AttributeValues.create(Dict.empty()))))
            .build();

    ModuleExtensionId fakeExtId =
        ModuleExtensionId.create(
            Label.parseCanonicalUnchecked("//:go_deps.bzl"), "go_deps", Optional.empty());

    BazelLockFileValue lockfile =
        BazelLockFileValue.builder()
            .setModuleExtensions(
                ImmutableMap.of(
                    fakeExtId,
                    ImmutableMap.of(
                        ModuleExtensionEvalFactors.create("", ""), fakeExtension)))
            .build();

    BzlmodExtensionInputsHelper helper = BzlmodExtensionInputsHelper.create(lockfile);
    var result = helper.getRecordedFilesForRepo("com_github_foo");

    assertThat(result).containsExactly(goModPath);
  }

  @Test
  public void getRecordedFilesForRepo_ignoresNonFileInputs() throws Exception {
    RepoRecordedInput.RepoCacheFriendlyPath goModPath =
        RepoRecordedInput.RepoCacheFriendlyPath.createInsideWorkspace(
            RepositoryName.MAIN, PathFragment.create("go.mod"));

    RepoRecordedInput.WithValue fileInput =
        new RepoRecordedInput.WithValue(new RepoRecordedInput.File(goModPath), "abc123");
    RepoRecordedInput.WithValue envInput =
        new RepoRecordedInput.WithValue(new RepoRecordedInput.EnvVar("GOPATH"), "/home/user/go");

    LockFileModuleExtension fakeExtension =
        LockFileModuleExtension.builder()
            .setBzlTransitiveDigest(new byte[0])
            .setUsagesDigest(new byte[0])
            .setRecordedInputs(ImmutableList.of(fileInput, envInput))
            .setGeneratedRepoSpecs(
                ImmutableMap.of(
                    "com_github_foo",
                    new RepoSpec(
                        new RepoRuleId(
                            Label.parseCanonicalUnchecked("//:ext.bzl"), "go_repository"),
                        AttributeValues.create(Dict.empty()))))
            .build();

    ModuleExtensionId fakeExtId =
        ModuleExtensionId.create(
            Label.parseCanonicalUnchecked("//:go_deps.bzl"), "go_deps", Optional.empty());

    BazelLockFileValue lockfile =
        BazelLockFileValue.builder()
            .setModuleExtensions(
                ImmutableMap.of(
                    fakeExtId,
                    ImmutableMap.of(
                        ModuleExtensionEvalFactors.create("", ""), fakeExtension)))
            .build();

    BzlmodExtensionInputsHelper helper = BzlmodExtensionInputsHelper.create(lockfile);
    var result = helper.getRecordedFilesForRepo("com_github_foo");

    // Only the File input should be returned, not the EnvVar
    assertThat(result).containsExactly(goModPath);
  }

  @Test
  public void getRecordedFilesForRepo_returnsEmpty_whenRepoNotFound() {
    BazelLockFileValue emptyLockfile = BazelLockFileValue.builder().build();

    BzlmodExtensionInputsHelper helper = BzlmodExtensionInputsHelper.create(emptyLockfile);
    var result = helper.getRecordedFilesForRepo("nonexistent_repo");

    assertThat(result).isEmpty();
  }

  @Test
  public void getRecordedFilesForRepo_returnsEmpty_whenExtensionHasNoFileInputs() throws Exception {
    LockFileModuleExtension fakeExtension =
        LockFileModuleExtension.builder()
            .setBzlTransitiveDigest(new byte[0])
            .setUsagesDigest(new byte[0])
            .setRecordedInputs(ImmutableList.of())
            .setGeneratedRepoSpecs(
                ImmutableMap.of(
                    "com_github_foo",
                    new RepoSpec(
                        new RepoRuleId(
                            Label.parseCanonicalUnchecked("//:ext.bzl"), "go_repository"),
                        AttributeValues.create(Dict.empty()))))
            .build();

    ModuleExtensionId fakeExtId =
        ModuleExtensionId.create(
            Label.parseCanonicalUnchecked("//:go_deps.bzl"), "go_deps", Optional.empty());

    BazelLockFileValue lockfile =
        BazelLockFileValue.builder()
            .setModuleExtensions(
                ImmutableMap.of(
                    fakeExtId,
                    ImmutableMap.of(
                        ModuleExtensionEvalFactors.create("", ""), fakeExtension)))
            .build();

    BzlmodExtensionInputsHelper helper = BzlmodExtensionInputsHelper.create(lockfile);
    var result = helper.getRecordedFilesForRepo("com_github_foo");

    assertThat(result).isEmpty();
  }
}
