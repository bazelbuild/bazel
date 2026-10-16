// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction.BazelLockfileFunctionException;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.gson.JsonSyntaxException;
import java.io.IOException;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelLockFileFunction}. */
@RunWith(JUnit4.class)
public final class BazelLockFileFunctionTest extends FoundationTestCase {
  private static BazelLockFileValue valueWithRegistryFile(String url) {
    return BazelLockFileValue.builder()
        .setRegistryFileHashes(ImmutableMap.of(unicodeToInternal(url), Optional.empty()))
        .build();
  }

  private RootedPath writeLockfile(String json) throws IOException {
    scratch.file("MODULE.bazel.lock", UTF_8, json);
    return RootedPath.toRootedPath(root, LabelConstants.MODULE_LOCKFILE_NAME);
  }

  @Test
  public void readsBazelWrittenLockfile() throws Exception {
    BazelLockFileValue expected =
        valueWithRegistryFile(
            "https://registry.example/modules/"
                + "x".repeat(8192)
                + "/café/日本語/😀/MODULE.bazel");
    BazelLockFileModule.updateLockfile(rootDirectory, expected);
    RootedPath path = RootedPath.toRootedPath(root, LabelConstants.MODULE_LOCKFILE_NAME);

    assertThat(FileSystemUtils.readContent(path.asPath(), UTF_8))
        .startsWith(BazelLockFileFunction.CURRENT_VERSION_HEADER);
    assertThat(BazelLockFileFunction.getLockfileValue(path, LockfileMode.UPDATE))
        .isEqualTo(expected);
    assertThat(BazelLockFileFunction.getLockfileValue(path, LockfileMode.ERROR)).isEqualTo(expected);
  }

  @Test
  public void findsVersionAfterLargeUnknownField() throws Exception {
    BazelLockFileValue expected =
        valueWithRegistryFile("https://registry.example/modules/b/2.0/MODULE.bazel");
    String canonicalJson = GsonTypeAdapterUtil.LOCKFILE_GSON.toJson(expected);
    RootedPath path =
        writeLockfile(
            "{\"unknown\": \""
                + "x".repeat(8193)
                + "\","
                + canonicalJson.substring(1));

    assertThat(BazelLockFileFunction.getLockfileValue(path, LockfileMode.UPDATE))
        .isEqualTo(expected);
  }

  @Test
  public void versionExtendingCurrentNumberIsNotCurrentVersion() throws Exception {
    String header = BazelLockFileFunction.CURRENT_VERSION_HEADER;
    RootedPath path =
        writeLockfile(
            header.substring(0, header.length() - 1) + "1,\"moduleExtensions\":{}}");

    assertThat(BazelLockFileFunction.getLockfileValue(path, LockfileMode.UPDATE))
        .isSameInstanceAs(BazelLockFileValue.EMPTY_LOCKFILE);
    BazelLockfileFunctionException exception =
        assertThrows(
            BazelLockfileFunctionException.class,
            () -> BazelLockFileFunction.getLockfileValue(path, LockfileMode.ERROR));
    assertThat(exception)
        .hasCauseThat()
        .hasMessageThat()
        .contains("The version of MODULE.bazel.lock is not supported");
  }

  @Test
  public void compactCurrentLockfileIsAccepted() throws Exception {
    BazelLockFileValue expected =
        valueWithRegistryFile("https://registry.example/modules/d/4.0/MODULE.bazel");
    String canonicalJson = GsonTypeAdapterUtil.LOCKFILE_GSON.toJson(expected);
    RootedPath path = writeLockfile(canonicalJson.replace("\n", "").replace("  ", ""));

    assertThat(BazelLockFileFunction.getLockfileValue(path, LockfileMode.ERROR)).isEqualTo(expected);
  }

  @Test
  public void unsupportedVersionSkipsMalformedBodyOrReportsVersionInErrorMode() throws Exception {
    RootedPath path =
        writeLockfile(
            "{\"lockFileVersion\": "
                + (BazelLockFileValue.LOCK_FILE_VERSION - 1)
                + ", broken_json");

    assertThat(BazelLockFileFunction.getLockfileValue(path, LockfileMode.UPDATE))
        .isSameInstanceAs(BazelLockFileValue.EMPTY_LOCKFILE);
    BazelLockfileFunctionException exception =
        assertThrows(
            BazelLockfileFunctionException.class,
            () -> BazelLockFileFunction.getLockfileValue(path, LockfileMode.ERROR));
    assertThat(exception)
        .hasCauseThat()
        .hasMessageThat()
        .contains("The version of MODULE.bazel.lock is not supported");
  }

  @Test
  public void supportedVersionRejectsTrailingJson() throws Exception {
    BazelLockFileValue value =
        valueWithRegistryFile("https://registry.example/modules/c/3.0/MODULE.bazel");
    RootedPath path = writeLockfile(GsonTypeAdapterUtil.LOCKFILE_GSON.toJson(value) + "\n{}");

    assertThrows(
        JsonSyntaxException.class,
        () -> BazelLockFileFunction.getLockfileValue(path, LockfileMode.UPDATE));
  }

  @Test
  public void missingLockfileIsEmptyEvenInErrorMode() throws Exception {
    RootedPath missing = RootedPath.toRootedPath(root, LabelConstants.MODULE_LOCKFILE_NAME);

    assertThat(BazelLockFileFunction.getLockfileValue(missing, LockfileMode.ERROR))
        .isSameInstanceAs(BazelLockFileValue.EMPTY_LOCKFILE);
  }
}
