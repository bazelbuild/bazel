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

import com.google.common.base.Strings;
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
    String fakeSha256 = Strings.repeat("a", 64);
    assertThat(repositoryCache.exists(fakeSha256, KeyType.SHA256)).isFalse();
  }

  /**
   * Test that the put method correctly stores the downloaded file into the cache.
   */
  @Test
  public void testPutCacheValue() throws IOException {
    repositoryCache.put(downloadedFileSha256, downloadedFile, KeyType.SHA256);

    Path cacheEntry = KeyType.SHA256.getCachePath(contentAddressableCachePath).getChild(downloadedFileSha256);
    Path cacheValue = cacheEntry.getChild(RepositoryCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /**
   * Test that the put mehtod without cache key correctly stores the downloaded file into the cache.
   */
  @Test
  public void testPutCacheValueWithoutHash() throws IOException {
    String cacheKey = repositoryCache.put(downloadedFile, KeyType.SHA256);
    assertThat(cacheKey).isEqualTo(downloadedFileSha256);

    Path cacheEntry =
        KeyType.SHA256.getCachePath(contentAddressableCachePath).getChild(downloadedFileSha256);
    Path cacheValue = cacheEntry.getChild(RepositoryCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /**
   * Test that the put method is idempotent, i.e. two successive put calls
   * should not affect the final state in the cache.
   */
  @Test
  public void testPutCacheValueIdempotent() throws IOException {
    repositoryCache.put(downloadedFileSha256, downloadedFile, KeyType.SHA256);
    repositoryCache.put(downloadedFileSha256, downloadedFile, KeyType.SHA256);

    Path cacheEntry = KeyType.SHA256.getCachePath(contentAddressableCachePath).getChild(downloadedFileSha256);
    Path cacheValue = cacheEntry.getChild(RepositoryCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /**
   * Test that the get method correctly retrieves the cached file from the cache.
   */
  @Test
  public void testGetCacheValue() throws IOException {
    // Inject file into cache
    repositoryCache.put(downloadedFileSha256, downloadedFile, KeyType.SHA256);

    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());
    Path actualTargetPath = repositoryCache.get(downloadedFileSha256, targetPath, KeyType.SHA256);

    // Check that the contents are the same.
    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(actualTargetPath, Charset.defaultCharset()));

    // Check that the returned value is stored under outputBaseExternal.
    assertThat((Object) actualTargetPath).isEqualTo(targetPath);
  }

  /**
   * Test that the get method retrieves a null if the value is not cached.
   */
  @Test
  public void testGetNullCacheValue() throws IOException {
    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());
    Path actualTargetPath = repositoryCache.get(downloadedFileSha256, targetPath, KeyType.SHA256);

    assertThat(actualTargetPath).isNull();
  }

  @Test
  public void testInvalidSha256Throws() throws IOException {
    String invalidSha = "foo";
    thrown.expect(IOException.class);
    thrown.expectMessage("Invalid key \"foo\" of type SHA-256");
    repositoryCache.put(invalidSha, downloadedFile, KeyType.SHA256);
  }

  @Test
  public void testPoisonedCache() throws IOException {
    Path poisonedEntry = KeyType.SHA256
        .getCachePath(contentAddressableCachePath).getChild(downloadedFileSha256);
    Path poisonedValue = poisonedEntry.getChild(RepositoryCache.DEFAULT_CACHE_FILENAME);
    scratch.file(poisonedValue.getPathString(), Charset.defaultCharset(), "poisoned");

    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());

    thrown.expect(IOException.class);
    thrown.expectMessage("does not match expected");
    thrown.expectMessage("Please delete the directory");

    repositoryCache.get(downloadedFileSha256, targetPath, KeyType.SHA256);
  }

  @Test
  public void testGetChecksum() throws IOException {
    String actualChecksum = RepositoryCache.getChecksum(KeyType.SHA256, downloadedFile);
    assertThat(actualChecksum).isEqualTo(downloadedFileSha256);
  }

  @Test
  public void testAssertFileChecksumPass() throws IOException {
    RepositoryCache.assertFileChecksum(downloadedFileSha256, downloadedFile, KeyType.SHA256);
  }

  @Test
  public void testAssertFileChecksumFail() throws IOException {
    thrown.expect(IOException.class);
    thrown.expectMessage("does not match expected");
    RepositoryCache.assertFileChecksum(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        downloadedFile,
        KeyType.SHA256);
  }

}
