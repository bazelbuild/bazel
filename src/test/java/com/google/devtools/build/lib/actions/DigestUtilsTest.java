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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Strings;
import com.google.common.cache.CacheStats;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.CheckReturnValue;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for DigestUtils.
 */
@RunWith(JUnit4.class)
public class DigestUtilsTest {

  @After
  public void tearDown() {
    DigestUtils.configureCache(0);
  }

  private static void assertDigestCalculationConcurrency(boolean expectConcurrent,
      final boolean fastDigest, final int fileSize1, final int fileSize2,
      HashFunction hf) throws Exception {
    final CountDownLatch barrierLatch = new CountDownLatch(2); // Used to block test threads.
    final CountDownLatch readyLatch = new CountDownLatch(1);   // Used to block main thread.

    FileSystem myfs =
        new InMemoryFileSystem(BlazeClock.instance(), hf) {
          @Override
          protected byte[] getDigest(Path path, HashFunction hashFunction) throws IOException {
            try {
              barrierLatch.countDown();
              readyLatch.countDown();
              // Either both threads will be inside getDigest at the same time or they
              // both will be blocked.
              barrierLatch.await();
            } catch (Exception e) {
              throw new IOException(e);
            }
            return super.getDigest(path, hashFunction);
          }

          @Override
          protected byte[] getFastDigest(Path path, HashFunction hashFunction) throws IOException {
            return fastDigest ? super.getDigest(path, hashFunction) : null;
          }
        };

    final Path myFile1 = myfs.getPath("/f1.dat");
    final Path myFile2 = myfs.getPath("/f2.dat");
    FileSystemUtils.writeContentAsLatin1(myFile1, Strings.repeat("a", fileSize1));
    FileSystemUtils.writeContentAsLatin1(myFile2, Strings.repeat("b", fileSize2));

     TestThread thread1 = new TestThread () {
       @Override public void runTest() throws Exception {
         DigestUtils.getDigestOrFail(myFile1, fileSize1);
       }
     };

     TestThread thread2 = new TestThread () {
       @Override public void runTest() throws Exception {
         DigestUtils.getDigestOrFail(myFile2, fileSize2);
       }
     };

     thread1.start();
     thread2.start();
     if (!expectConcurrent) { // Synchronized case.
      // Wait until at least one thread reached getDigest().
      assertThat(readyLatch.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
      // Only 1 thread should be inside getDigest().
      assertThat(barrierLatch.getCount()).isEqualTo(1);
       barrierLatch.countDown(); // Release barrier latch, allowing both threads to proceed.
     }
     // Test successful execution within 5 seconds.
     thread1.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
     thread2.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
  }

  /**
   * Ensures that digest calculation is synchronized for files
   * greater than 4096 bytes if the digest is not available cheaply,
   * so machines with rotating drives don't become unusable.
   */
  @Test
  public void testCalculationConcurrency() throws Exception {
    for (HashFunction hf : Arrays.asList(HashFunction.MD5, HashFunction.SHA1)) {
      assertDigestCalculationConcurrency(true, true, 4096, 4096, hf);
      assertDigestCalculationConcurrency(true, true, 4097, 4097, hf);
      assertDigestCalculationConcurrency(true, false, 4096, 4096, hf);
      assertDigestCalculationConcurrency(false, false, 4097, 4097, hf);
      assertDigestCalculationConcurrency(true, false, 1024, 4097, hf);
      assertDigestCalculationConcurrency(true, false, 1024, 1024, hf);
    }
  }

  public void assertRecoverFromMalformedDigest(HashFunction... hashFunctions) throws Exception {
    for (HashFunction hf : hashFunctions) {
      final byte[] malformed = {0, 0, 0};
      FileSystem myFS =
          new InMemoryFileSystem(BlazeClock.instance(), hf) {
            @Override
            protected byte[] getFastDigest(Path path, HashFunction hashFunction)
                throws IOException {
              // Digest functions have more than 3 bytes, usually at least 16.
              return malformed;
            }
          };
      Path path = myFS.getPath("/file");
      FileSystemUtils.writeContentAsLatin1(path, "a");
      byte[] result = DigestUtils.getDigestOrFail(path, 1);
      assertThat(result).isEqualTo(path.getDigest());
      assertThat(result).isNotSameAs(malformed);
      assertThat(path.isValidDigest(result)).isTrue();
    }
  }

  @Test
  public void testRecoverFromMalformedDigestWithoutCache() throws Exception {
    try {
      DigestUtils.getCacheStats();
      fail("Digests cache should remain disabled until configureCache is called");
    } catch (NullPointerException expected) {
    }
    assertRecoverFromMalformedDigest(HashFunction.MD5, HashFunction.SHA1);
    try {
      DigestUtils.getCacheStats();
      fail("Digests cache was unexpectedly enabled through the test");
    } catch (NullPointerException expected) {
    }
  }

  @Test
  public void testRecoverFromMalformedDigestWithCache() throws Exception {
    DigestUtils.configureCache(10);
    assertThat(DigestUtils.getCacheStats()).isNotNull(); // Ensure the cache is enabled.

    // When using the cache, we cannot run our test using different hash functions because the
    // hash function is not part of the cache key. This is intentional: the hash function is
    // essentially final and can only be changed for tests. Therefore, just test the same hash
    // function twice to further exercise the cache code.
    assertRecoverFromMalformedDigest(HashFunction.MD5, HashFunction.MD5);

    assertThat(DigestUtils.getCacheStats()).isNotNull(); // Ensure the cache remains enabled.
  }

  /** Helper class to assert the cache statistics. */
  private static class CacheStatsChecker {
    /** Cache statistics, grabbed at construction time. */
    private final CacheStats stats;

    private int expectedEvictionCount;
    private int expectedHitCount;
    private int expectedMissCount;

    CacheStatsChecker() {
      this.stats = DigestUtils.getCacheStats();
    }

    @CheckReturnValue
    CacheStatsChecker evictionCount(int count) {
      expectedEvictionCount = count;
      return this;
    }

    @CheckReturnValue
    CacheStatsChecker hitCount(int count) {
      expectedHitCount = count;
      return this;
    }

    @CheckReturnValue
    CacheStatsChecker missCount(int count) {
      expectedMissCount = count;
      return this;
    }

    void check() throws Exception {
      assertThat(stats.evictionCount()).isEqualTo(expectedEvictionCount);
      assertThat(stats.hitCount()).isEqualTo(expectedHitCount);
      assertThat(stats.missCount()).isEqualTo(expectedMissCount);
    }
  }

  @Test
  public void testCache() throws Exception {
    final AtomicInteger getFastDigestCounter = new AtomicInteger(0);
    final AtomicInteger getDigestCounter = new AtomicInteger(0);

    FileSystem tracingFileSystem =
        new InMemoryFileSystem(BlazeClock.instance()) {
          @Override
          protected byte[] getFastDigest(Path path, HashFunction hashFunction) throws IOException {
            getFastDigestCounter.incrementAndGet();
            return null;
          }

          @Override
          protected byte[] getDigest(Path path, HashFunction hashFunction) throws IOException {
            getDigestCounter.incrementAndGet();
            return super.getDigest(path, hashFunction);
          }
        };

    DigestUtils.configureCache(2);

    final Path file1 = tracingFileSystem.getPath("/1.txt");
    final Path file2 = tracingFileSystem.getPath("/2.txt");
    final Path file3 = tracingFileSystem.getPath("/3.txt");
    FileSystemUtils.writeContentAsLatin1(file1, "some contents");
    FileSystemUtils.writeContentAsLatin1(file2, "some other contents");
    FileSystemUtils.writeContentAsLatin1(file3, "and something else");

    byte[] digest1 = DigestUtils.getDigestOrFail(file1, file1.getFileSize());
    assertThat(getFastDigestCounter.get()).isEqualTo(1);
    assertThat(getDigestCounter.get()).isEqualTo(1);
    new CacheStatsChecker().evictionCount(0).hitCount(0).missCount(1).check();

    byte[] digest2 = DigestUtils.getDigestOrFail(file1, file1.getFileSize());
    assertThat(getFastDigestCounter.get()).isEqualTo(2);
    assertThat(getDigestCounter.get()).isEqualTo(1);
    new CacheStatsChecker().evictionCount(0).hitCount(1).missCount(1).check();

    assertThat(digest2).isEqualTo(digest1);

    // Evict the digest for the previous file.
    DigestUtils.getDigestOrFail(file2, file2.getFileSize());
    DigestUtils.getDigestOrFail(file3, file3.getFileSize());
    new CacheStatsChecker().evictionCount(1).hitCount(1).missCount(3).check();

    // And now try to recompute it.
    byte[] digest3 = DigestUtils.getDigestOrFail(file1, file1.getFileSize());
    new CacheStatsChecker().evictionCount(2).hitCount(1).missCount(4).check();

    assertThat(digest3).isEqualTo(digest1);
  }
}
