// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.cache;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;

import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.nio.charset.Charset;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link RepositoryCache}.
 */
@RunWith(JUnit4.class)
public class RepositoryCacheTest {

  @Rule public ExpectedException thrown = ExpectedException.none();

  private Scratch scratch;
  private RepositoryCache repositoryCache;
  private Path repositoryCachePath;
  private Path contentAddressableCachePath;
  private Path downloadedFile;
  private String downloadedFileSha256;

  @Before
  public void setUp() throws Exception {
    scratch = new Scratch("/");
    repositoryCachePath = scratch.dir("/repository_cache");
    repositoryCache = new RepositoryCache();
    repositoryCache.setRepositoryCachePath(repositoryCachePath);
    contentAddressableCachePath = repositoryCache.getContentAddressableCachePath();

    downloadedFile = scratch.file("file.tmp", Charset.defaultCharset(), "contents");
    downloadedFileSha256 = "bfe5ed57e6e323555b379c660aa8d35b70c2f8f07cf03ad6747266495ac13be0";
  }

  @After
  public void tearDown() throws IOException {
    repositoryCachePath.deleteTree();
  }

  @Test
  public void testNonExistentCacheValue() {
    String fakeSha256 = "a".repeat(64);
    assertThat(repositoryCache.exists(fakeSha256, KeyType.SHA256)).isFalse();
  }

  /** Test that the put method correctly stores the downloaded file into the cache. */
  @Test
  public void testPutCacheValue() throws Exception {
    repositoryCache.put(
        downloadedFileSha256, downloadedFile, KeyType.SHA256, /* canonicalId= */ null);

    Path cacheEntry = KeyType.SHA256.getCachePath(contentAddressableCachePath).getChild(downloadedFileSha256);
    Path cacheValue = cacheEntry.getChild(RepositoryCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /**
   * Test that the put method without cache key correctly stores the downloaded file into the cache.
   */
  @Test
  public void testPutCacheValueWithoutHash() throws Exception {
    String cacheKey = repositoryCache.put(downloadedFile, KeyType.SHA256, /* canonicalId= */ null);
    assertThat(cacheKey).isEqualTo(downloadedFileSha256);

    Path cacheEntry =
        KeyType.SHA256.getCachePath(contentAddressableCachePath).getChild(downloadedFileSha256);
    Path cacheValue = cacheEntry.getChild(RepositoryCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /**
   * Test that the put method is idempotent, i.e. two successive put calls should not affect the
   * final state in the cache.
   */
  @Test
  public void testPutCacheValueIdempotent() throws Exception {
    repositoryCache.put(
        downloadedFileSha256, downloadedFile, KeyType.SHA256, /* canonicalId= */ null);
    repositoryCache.put(
        downloadedFileSha256, downloadedFile, KeyType.SHA256, /* canonicalId= */ null);

    Path cacheEntry = KeyType.SHA256.getCachePath(contentAddressableCachePath).getChild(downloadedFileSha256);
    Path cacheValue = cacheEntry.getChild(RepositoryCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /** Test that the get method correctly retrieves the cached file from the cache. */
  @Test
  public void testGetCacheValue() throws Exception {
    // Inject file into cache
    repositoryCache.put(
        downloadedFileSha256, downloadedFile, KeyType.SHA256, /* canonicalId= */ null);

    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());
    Path actualTargetPath =
        repositoryCache.get(
            downloadedFileSha256, targetPath, KeyType.SHA256, /* canonicalId= */ null);

    // Check that the contents are the same.
    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(actualTargetPath, Charset.defaultCharset()));

    // Check that the returned value is stored under outputBaseExternal.
    assertThat((Object) actualTargetPath).isEqualTo(targetPath);
  }

  /** Test that the get method retrieves a null if the value is not cached. */
  @Test
  public void testGetNullCacheValue() throws Exception {
    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());
    Path actualTargetPath =
        repositoryCache.get(
            downloadedFileSha256, targetPath, KeyType.SHA256, /* canonicalId= */ null);

    assertThat(actualTargetPath).isNull();
  }

  @Test
  public void testInvalidSha256Throws() throws Exception {
    String invalidSha = "foo";
    thrown.expect(IOException.class);
    thrown.expectMessage("Invalid key \"foo\" of type SHA-256");
    repositoryCache.put(invalidSha, downloadedFile, KeyType.SHA256, /* canonicalId= */ null);
  }

  @Test
  public void testPoisonedCache() throws Exception {
    Path poisonedEntry = KeyType.SHA256
        .getCachePath(contentAddressableCachePath).getChild(downloadedFileSha256);
    Path poisonedValue = poisonedEntry.getChild(RepositoryCache.DEFAULT_CACHE_FILENAME);
    scratch.file(poisonedValue.getPathString(), Charset.defaultCharset(), "poisoned");

    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());

    thrown.expect(IOException.class);
    thrown.expectMessage("does not match expected");
    thrown.expectMessage("Please delete the directory");

    repositoryCache.get(downloadedFileSha256, targetPath, KeyType.SHA256, /* canonicalId= */ null);
  }

  @Test
  public void testGetChecksum() throws Exception {
    String actualChecksum = RepositoryCache.getChecksum(KeyType.SHA256, downloadedFile);
    assertThat(actualChecksum).isEqualTo(downloadedFileSha256);
  }

  @Test
  public void testGetChecksumWithFastDigest() throws Exception {
    String fastDigestChecksum = "cfe5ed57e6e323555b379c660aa8d35b70c2f8f07cf03ad6747266495ac13be0";
    downloadedFile = spy(downloadedFile);
    doReturn(BaseEncoding.base16().lowerCase().decode(fastDigestChecksum))
        .when(downloadedFile)
        .getFastDigest();

    String actualChecksum = RepositoryCache.getChecksum(KeyType.SHA256, downloadedFile);
    assertThat(actualChecksum).isEqualTo(fastDigestChecksum);
  }

  @Test
  public void testAssertFileChecksumPass() throws Exception {
    RepositoryCache.assertFileChecksum(downloadedFileSha256, downloadedFile, KeyType.SHA256);
  }

  @Test
  public void testAssertFileChecksumFail() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("does not match expected");
    RepositoryCache.assertFileChecksum(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        downloadedFile,
        KeyType.SHA256);
  }

  @Test
  public void testCanonicalId() throws Exception {
    repositoryCache.put(downloadedFileSha256, downloadedFile, KeyType.SHA256, "fooid");
    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());

    Path lookupWithSameId =
        repositoryCache.get(downloadedFileSha256, targetPath, KeyType.SHA256, "fooid");
    assertThat(lookupWithSameId).isEqualTo(targetPath);

    Path lookupOtherId =
        repositoryCache.get(downloadedFileSha256, targetPath, KeyType.SHA256, "barid");
    assertThat(lookupOtherId).isNull();

    Path lookupNoId =
        repositoryCache.get(
            downloadedFileSha256, targetPath, KeyType.SHA256, /* canonicalId= */ null);
    assertThat(lookupNoId).isEqualTo(targetPath);
  }
}
