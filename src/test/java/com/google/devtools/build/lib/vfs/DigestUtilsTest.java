// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.HexFormat;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link DigestUtils}. */
@RunWith(TestParameterInjector.class)
public final class DigestUtilsTest {

  @After
  public void tearDown() {
    DigestUtils.configureCache(/*maximumSize=*/ 0);
  }

  @Test
  public void testCache() throws Exception {
    AtomicInteger getFastDigestCounter = new AtomicInteger(0);
    AtomicInteger getDigestCounter = new AtomicInteger(0);

    FileSystem tracingFileSystem =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public byte[] getFastDigest(PathFragment path) {
            getFastDigestCounter.incrementAndGet();
            return null;
          }

          @Override
          public byte[] getDigest(PathFragment path) throws IOException {
            getDigestCounter.incrementAndGet();
            return super.getDigest(path);
          }
        };

    DigestUtils.configureCache(/*maximumSize=*/ 100);

    Path file = tracingFileSystem.getPath("/file.txt");
    FileSystemUtils.writeContentAsLatin1(file, "some contents");

    byte[] digest =
        DigestUtils.getDigestWithManualFallback(file, SyscallCache.NO_CACHE, /* status= */ null);
    assertThat(getFastDigestCounter.get()).isEqualTo(1);
    assertThat(getDigestCounter.get()).isEqualTo(1);

    assertThat(
            DigestUtils.getDigestWithManualFallback(
                file, SyscallCache.NO_CACHE, /* status= */ null))
        .isEqualTo(digest);
    assertThat(getFastDigestCounter.get()).isEqualTo(2);
    assertThat(getDigestCounter.get()).isEqualTo(1); // Cached.

    DigestUtils.clearCache();

    assertThat(
            DigestUtils.getDigestWithManualFallback(
                file, SyscallCache.NO_CACHE, /* status= */ null))
        .isEqualTo(digest);
    assertThat(getFastDigestCounter.get()).isEqualTo(3);
    assertThat(getDigestCounter.get()).isEqualTo(2); // Not cached.
  }

  @Test
  public void cacheHitWithoutRestatingWhenTheCallerSuppliesTheStat() throws Exception {
    AtomicInteger getDigestCounter = new AtomicInteger(0);
    AtomicInteger statCounter = new AtomicInteger(0);

    FileSystem tracingFileSystem =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public byte[] getDigest(PathFragment path) throws IOException {
            getDigestCounter.incrementAndGet();
            return super.getDigest(path);
          }

          @Override
          public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
            statCounter.incrementAndGet();
            return super.stat(path, followSymlinks);
          }
        };

    DigestUtils.configureCache(/* maximumSize= */ 100);

    Path file = tracingFileSystem.getPath("/file.txt");
    FileSystemUtils.writeContentAsLatin1(file, "some contents");

    // Without a stat, DigestUtils has to stat the file itself to build the cache key.
    byte[] digest = DigestUtils.manuallyComputeDigest(file, /* status= */ null);
    assertThat(getDigestCounter.get()).isEqualTo(1);
    assertThat(statCounter.get()).isEqualTo(1);

    // A caller that already stat'ed the file gets the cached digest without a further stat, which
    // only works if both spellings of the key agree.
    FileStatus stat = file.stat();
    getDigestCounter.set(0);
    statCounter.set(0);
    assertThat(DigestUtils.manuallyComputeDigest(file, stat)).isEqualTo(digest);
    assertThat(getDigestCounter.get()).isEqualTo(0);
    assertThat(statCounter.get()).isEqualTo(0);
  }

  @Test
  public void modifiedFile_staleEntryIsAMissAndIsReplaced() throws Exception {
    AtomicInteger getDigestCounter = new AtomicInteger(0);
    FileSystem tracingFileSystem =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public byte[] getDigest(PathFragment path) throws IOException {
            getDigestCounter.incrementAndGet();
            return super.getDigest(path);
          }
        };

    DigestUtils.configureCache(/* maximumSize= */ 100);

    Path file = tracingFileSystem.getPath("/file.txt");
    FileSystemUtils.writeContentAsLatin1(file, "some contents");
    byte[] digest = DigestUtils.manuallyComputeDigest(file, /* status= */ null);
    assertThat(getDigestCounter.get()).isEqualTo(1);
    assertThat(DigestUtils.getCacheStats().missCount()).isEqualTo(1);

    // The file changes (in size, so no clock is needed). The entry is present but stale: that is a
    // miss, the digest is recomputed, and the entry is replaced rather than kept alongside.
    FileSystemUtils.writeContentAsLatin1(file, "other contents!");
    byte[] newDigest = DigestUtils.manuallyComputeDigest(file, /* status= */ null);
    assertThat(newDigest).isNotEqualTo(digest);
    assertThat(getDigestCounter.get()).isEqualTo(2);
    assertThat(DigestUtils.getCacheStats().missCount()).isEqualTo(2);
    assertThat(DigestUtils.getCacheStats().hitCount()).isEqualTo(0);
    assertThat(DigestUtils.getCacheStats().evictionCount()).isEqualTo(0);

    // The replaced entry serves the new contents without a further read.
    assertThat(DigestUtils.manuallyComputeDigest(file, /* status= */ null)).isEqualTo(newDigest);
    assertThat(getDigestCounter.get()).isEqualTo(2);
    assertThat(DigestUtils.getCacheStats().hitCount()).isEqualTo(1);
  }

  @Test
  public void manuallyComputeDigest() throws Exception {
    byte[] digest = {1, 2, 3};
    FileSystem noDigestFileSystem =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public byte[] getFastDigest(PathFragment path) {
            throw new AssertionError("Unexpected call to getFastDigest");
          }

          @Override
          public byte[] getDigest(PathFragment path) {
            return digest;
          }
        };
    Path file = noDigestFileSystem.getPath("/f.txt");
    FileSystemUtils.writeContentAsLatin1(file, "contents");

    assertThat(DigestUtils.manuallyComputeDigest(file, /* status= */ null)).isEqualTo(digest);
  }

  @Test
  public void combineUnordered_commutative() {
    byte[] a = {1, 2, 3};
    byte[] b = {4, 5, 6};
    assertThat(DigestUtils.combineUnordered(a.clone(), b.clone()))
        .isEqualTo(DigestUtils.combineUnordered(b.clone(), a.clone()));
  }

  @Test
  public void combineUnordered_noCancellation() {
    byte[] a = {1, 2, 3};
    assertThat(DigestUtils.combineUnordered(a.clone(), a.clone()))
        .isNotEqualTo(new byte[] {0, 0, 0});
  }

  @Test
  public void toHexByteString_matchesLowercaseHexEncoding(
      @TestParameter({
            "",
            "00",
            "0f",
            "10",
            "7f",
            "80",
            "ff",
            "0123456789abcdef",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
          })
          String hex) {
    assertThat(DigestUtils.toHexByteString(HexFormat.of().parseHex(hex)).toStringUtf8())
        .isEqualTo(hex);
  }
}
