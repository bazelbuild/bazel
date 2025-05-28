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
import com.google.devtools.build.lib.bazel.repository.cache.DownloadCache.KeyType;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.bazel.BazelHashFunctions;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests for {@link DownloadCache}. */
@RunWith(Parameterized.class)
public class DownloadCacheTest {

  @Rule public ExpectedException thrown = ExpectedException.none();

  private Scratch scratch;
  private DownloadCache downloadCache;
  private Path repositoryCachePath;
  private Path downloadedFile;

  private final DigestHashFunction digestHashFunction;
  private final KeyType keyType;
  private final String hash;

  @Parameters
  public static List<Object[]> getKeyType() {
    List<Object[]> keyTypes = new ArrayList<>();
    keyTypes.add(
        new Object[] {
          // digestHashFunction
          DigestHashFunction.SHA256,
          // keyType
          KeyType.SHA256,
          // hash
          "bfe5ed57e6e323555b379c660aa8d35b70c2f8f07cf03ad6747266495ac13be0",
          // echo 'contents' | sha256sum
        });
    if (TestConstants.BLAKE3_AVAILABLE) {
      keyTypes.add(
          new Object[] {
            // digestHashFunction
            BazelHashFunctions.BLAKE3,
            // keyType
            KeyType.BLAKE3,
            // hash
            "54e00265e2516f168096da17059a6109563d9ba64a0b77cdc4b33e44600c2a39",
            // echo 'contents' | b3sum
          });
    }
    return keyTypes;
  }

  public DownloadCacheTest(DigestHashFunction digestHashFunction, KeyType keyType, String hash) {
    this.digestHashFunction = digestHashFunction;
    this.keyType = keyType;
    this.hash = hash;
  }

  @Before
  public void setUp() throws Exception {
    scratch = new Scratch("/");
    repositoryCachePath = scratch.dir("/repository_cache");
    downloadCache = new DownloadCache();
    downloadCache.setPath(repositoryCachePath);

    downloadedFile = scratch.file("file.tmp", Charset.defaultCharset(), "contents");
  }

  @After
  public void tearDown() throws IOException {
    repositoryCachePath.deleteTree();
  }

  @Test
  public void testNonExistentCacheValue() {
    String fakeHash = "a".repeat(64);
    assertThat(downloadCache.exists(fakeHash, keyType)).isFalse();
  }

  /** Test that the put method correctly stores the downloaded file into the cache. */
  @Test
  public void testPutCacheValue() throws Exception {
    downloadCache.put(hash, downloadedFile, keyType, /* canonicalId= */ null);

    Path cacheEntry = keyType.getCachePath(repositoryCachePath).getChild(hash);
    Path cacheValue = cacheEntry.getChild(DownloadCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /**
   * Test that the put method without cache key correctly stores the downloaded file into the cache.
   */
  @Test
  public void testPutCacheValueWithoutHash() throws Exception {
    String cacheKey = downloadCache.put(downloadedFile, keyType, /* canonicalId= */ null);
    assertThat(cacheKey).isEqualTo(hash);

    Path cacheEntry = keyType.getCachePath(repositoryCachePath).getChild(hash);
    Path cacheValue = cacheEntry.getChild(DownloadCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /**
   * Test that the put method is idempotent, i.e. two successive put calls should not affect the
   * final state in the cache.
   */
  @Test
  public void testPutCacheValueIdempotent() throws Exception {
    downloadCache.put(hash, downloadedFile, keyType, /* canonicalId= */ null);
    downloadCache.put(hash, downloadedFile, keyType, /* canonicalId= */ null);

    Path cacheEntry = keyType.getCachePath(repositoryCachePath).getChild(hash);
    Path cacheValue = cacheEntry.getChild(DownloadCache.DEFAULT_CACHE_FILENAME);

    assertThat(FileSystemUtils.readContent(downloadedFile, Charset.defaultCharset()))
        .isEqualTo(FileSystemUtils.readContent(cacheValue, Charset.defaultCharset()));
  }

  /** Test that the get method correctly retrieves the cached file from the cache. */
  @Test
  public void testGetCacheValue() throws Exception {
    // Inject file into cache
    downloadCache.put(hash, downloadedFile, keyType, /* canonicalId= */ null);

    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());
    Path actualTargetPath = downloadCache.get(hash, targetPath, keyType, /* canonicalId= */ null);

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
    Path actualTargetPath = downloadCache.get(hash, targetPath, keyType, /* canonicalId= */ null);

    assertThat(actualTargetPath).isNull();
  }

  @Test
  public void testInvalidSha256Throws() throws Exception {
    String invalidSha = "foo";
    thrown.expect(IOException.class);
    thrown.expectMessage("Invalid key \"foo\" of type " + keyType);
    downloadCache.put(invalidSha, downloadedFile, keyType, /* canonicalId= */ null);
  }

  @Test
  public void testPoisonedCache() throws Exception {
    Path poisonedEntry = keyType.getCachePath(repositoryCachePath).getChild(hash);
    Path poisonedValue = poisonedEntry.getChild(DownloadCache.DEFAULT_CACHE_FILENAME);
    scratch.file(poisonedValue.getPathString(), Charset.defaultCharset(), "poisoned");

    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());

    thrown.expect(IOException.class);
    thrown.expectMessage("does not match expected");
    thrown.expectMessage("Please delete the directory");

    downloadCache.get(hash, targetPath, keyType, /* canonicalId= */ null);
  }

  @Test
  public void testGetChecksum() throws Exception {
    String actualChecksum = DownloadCache.getChecksum(keyType, downloadedFile);
    assertThat(actualChecksum).isEqualTo(hash);
  }

  @Test
  public void testGetChecksumWithFastDigest() throws Exception {
    var fastDigestChecksum = "cfe5ed57e6e323555b379c660aa8d35b70c2f8f07cf03ad6747266495ac13be0";
    var fs = new InMemoryFileSystem(new JavaClock(), digestHashFunction);
    downloadedFile = spy(downloadedFile);
    doReturn(BaseEncoding.base16().lowerCase().decode(fastDigestChecksum))
        .when(downloadedFile)
        .getFastDigest();
    doReturn(fs).when(downloadedFile).getFileSystem();

    String actualChecksum = DownloadCache.getChecksum(keyType, downloadedFile);
    assertThat(actualChecksum).isEqualTo(fastDigestChecksum);
  }

  @Test
  public void testAssertFileChecksumPass() throws Exception {
    DownloadCache.assertFileChecksum(hash, downloadedFile, keyType);
  }

  @Test
  public void testAssertFileChecksumFail() throws Exception {
    thrown.expect(IOException.class);
    thrown.expectMessage("does not match expected");
    DownloadCache.assertFileChecksum(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        downloadedFile,
        keyType);
  }

  @Test
  public void testCanonicalId() throws Exception {
    downloadCache.put(hash, downloadedFile, keyType, "fooid");
    Path targetDirectory = scratch.dir("/external");
    Path targetPath = targetDirectory.getChild(downloadedFile.getBaseName());

    Path lookupWithSameId = downloadCache.get(hash, targetPath, keyType, "fooid");
    assertThat(lookupWithSameId).isEqualTo(targetPath);

    Path lookupOtherId = downloadCache.get(hash, targetPath, keyType, "barid");
    assertThat(lookupOtherId).isNull();

    Path lookupNoId = downloadCache.get(hash, targetPath, keyType, /* canonicalId= */ null);
    assertThat(lookupNoId).isEqualTo(targetPath);
  }
}
