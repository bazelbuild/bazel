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
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import java.io.IOException;
import java.net.URI;
import java.util.Map;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for git_repository remote and commit validation in IndexRegistry. */
@RunWith(JUnit4.class)
public final class IndexRegistryGitRepositoryTest extends FoundationTestCase {
  /**
   * Constructs a {@link DownloadManager} that returns the source.json unconditionally.
   * @param sourceJson The source.json the {@link DownloadManager} should return.
   */
  private DownloadManager mockDownloadManager(String sourceJson) throws Exception {
    return new DownloadManager(new DownloadCache(), null, null, reporter) {
      @Override
      public byte[] downloadAndReadOneUrlForBzlmod(
              URI url, Map<String, String> clientEnv, Optional<Checksum> checksum) {
        return sourceJson.getBytes(UTF_8);
      }
    };
  }

  /** Constructs a mock {@link Registry} with dummy information. */
  private Registry mockRegistry() throws Exception {
    return new RegistryFactoryImpl(Suppliers.ofInstance(ImmutableMap.of()))
            .createRegistry(
                    "https://fake.registry",
                    LockfileMode.UPDATE,
                    ImmutableMap.of(),
                    ImmutableMap.of(),
                    Optional.empty(),
                    ImmutableSet.of());
  }

  /** Constructs a {@link ModuleKey} for test_module with dummy information. */
  private ModuleKey mockModuleKey() throws Exception {
    return createModuleKey("test_module", "1.0");
  }

  /** Computes the known hashes for test_module with a dummy MODULE.bazel. */
  private ImmutableMap<String, Optional<Checksum>> createKnownHashes() throws Exception {
    return ImmutableMap.of(
            "https://fake.registry/modules/test_module/1.0/MODULE.bazel",
            Optional.of(
                    Checksum.fromString(
                            DownloadCache.KeyType.SHA256,
                            Hashing.sha256()
                                    .hashString("module(name = \"test_module\", version = \"1.0\")", UTF_8)
                                    .toString())));
  }

  private void assertValidSourceJson(String sourceJson) throws Exception {
    DownloadManager downloadManager = mockDownloadManager(sourceJson);
    Registry registry = mockRegistry();
    ModuleKey key = mockModuleKey();
    ImmutableMap<String, Optional<Checksum>> knownHashes = createKnownHashes();

    // Fetch the spec, and discard it.
    registry.getRepoSpec(key, knownHashes, reporter, downloadManager);
  }

  private void assertInvalidSourceJson(String sourceJson, String expectedMessageSubstring)
      throws Exception {
    DownloadManager downloadManager = mockDownloadManager(sourceJson);
    Registry registry = mockRegistry();
    ModuleKey key = mockModuleKey();
    ImmutableMap<String, Optional<Checksum>> knownHashes = createKnownHashes();

    IOException e =
        assertThrows(
            IOException.class,
            () -> registry.getRepoSpec(key, knownHashes, reporter, downloadManager));
    assertThat(e).hasMessageThat().contains(expectedMessageSubstring);
  }

  @Test
  public void getRepoSpec_invalidGitRepositoryAttributes_throws() throws Exception {
    assertInvalidSourceJson(
        """
        {
          "type": "git_repository",
          "remote": "ftp://example.com/repo.git",
          "commit": "0123456789abcdef0123456789abcdef01234567"
        }
        """,
        "Invalid remote URL scheme");

    assertInvalidSourceJson(
        """
        {
          "type": "git_repository",
          "remote": "https://example.com/repo.git",
          "commit": "invalid commit hash with spaces"
        }
        """,
        "Invalid commit");

    assertInvalidSourceJson(
        """
        {
          "type": "git_repository",
          "remote": "https://example.com/repo.git",
          "tag": "-invalid-tag"
        }
        """,
        "Invalid tag");
  }

  @Test
  public void getRepoSpec_validGitRepositoryAttributes_doesNotThrow() throws Exception {
    assertValidSourceJson(
        """
        {
          "type": "git_repository",
          "remote": "git@example.com:repo.git",
          "commit": "0123456789abcdef0123456789abcdef01234567"
        }
        """);

    assertValidSourceJson(
        """
        {
          "type": "git_repository",
          "remote": "ssh://git@example.com:repo.git",
          "commit": "0123456789abcdef0123456789abcdef01234567"
        }
        """);

    assertValidSourceJson(
        """
        {
          "type": "git_repository",
          "remote": "https://example.com/repo.git",
          "commit": "0123456789abcdef0123456789abcdef01234567"
        }
        """);
  }
}
