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

import com.google.common.base.Strings;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DigestUtils}. */
@RunWith(JUnit4.class)
public final class DigestUtilsTest {

  @After
  public void tearDown() {
    DigestUtils.configureCache(/*maximumSize=*/ 0);
  }

  private static void assertDigestCalculationConcurrency(
      boolean expectConcurrent,
      boolean fastDigest,
      int fileSize1,
      int fileSize2,
      DigestHashFunction hf)
      throws Exception {
    CountDownLatch barrierLatch = new CountDownLatch(2); // Used to block test threads.
    CountDownLatch readyLatch = new CountDownLatch(1); // Used to block main thread.

    FileSystem myfs =
        new InMemoryFileSystem(hf) {
          @Override
          protected byte[] getDigest(PathFragment path) throws IOException {
            try {
              barrierLatch.countDown();
              readyLatch.countDown();
              // Either both threads will be inside getDigest at the same time or they
              // both will be blocked.
              barrierLatch.await();
            } catch (Exception e) {
              throw new IOException(e);
            }
            return super.getDigest(path);
          }

          @Override
          protected byte[] getFastDigest(PathFragment path) throws IOException {
            return fastDigest ? super.getDigest(path) : null;
          }
        };

    Path myFile1 = myfs.getPath("/f1.dat");
    Path myFile2 = myfs.getPath("/f2.dat");
    FileSystemUtils.writeContentAsLatin1(myFile1, Strings.repeat("a", fileSize1));
    FileSystemUtils.writeContentAsLatin1(myFile2, Strings.repeat("b", fileSize2));

    TestThread thread1 =
        new TestThread(() -> DigestUtils.getDigestWithManualFallback(myFile1, fileSize1));
    TestThread thread2 =
        new TestThread(() -> DigestUtils.getDigestWithManualFallback(myFile2, fileSize2));
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
   * Ensures that digest calculation is synchronized for files greater than
   * {@link DigestUtils#MULTI_THREADED_DIGEST_MAX_FILE_SIZE} bytes if the digest is not
   * available cheaply, so machines with rotating drives don't become unusable.
   */
  @Test
  public void testCalculationConcurrency() throws Exception {
    int small = DigestUtils.MULTI_THREADED_DIGEST_MAX_FILE_SIZE;
    int large = DigestUtils.MULTI_THREADED_DIGEST_MAX_FILE_SIZE + 1;
    for (DigestHashFunction hf :
        Arrays.asList(DigestHashFunction.SHA256, DigestHashFunction.SHA1)) {
      assertDigestCalculationConcurrency(true, true, small, small, hf);
      assertDigestCalculationConcurrency(true, true, large, large, hf);
      assertDigestCalculationConcurrency(true, false, small, small, hf);
      assertDigestCalculationConcurrency(true, false, small, large, hf);
      assertDigestCalculationConcurrency(false, false, large, large, hf);
    }
  }

  @Test
  public void testCache() throws Exception {
    AtomicInteger getFastDigestCounter = new AtomicInteger(0);
    AtomicInteger getDigestCounter = new AtomicInteger(0);

    FileSystem tracingFileSystem =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          protected byte[] getFastDigest(PathFragment path) {
            getFastDigestCounter.incrementAndGet();
            return null;
          }

          @Override
          protected byte[] getDigest(PathFragment path) throws IOException {
            getDigestCounter.incrementAndGet();
            return super.getDigest(path);
          }
        };

    DigestUtils.configureCache(/*maximumSize=*/ 100);

    Path file = tracingFileSystem.getPath("/file.txt");
    FileSystemUtils.writeContentAsLatin1(file, "some contents");

    byte[] digest1 = DigestUtils.getDigestWithManualFallback(file, file.getFileSize());
    assertThat(getFastDigestCounter.get()).isEqualTo(1);
    assertThat(getDigestCounter.get()).isEqualTo(1);

    assertThat(DigestUtils.getDigestWithManualFallback(file, file.getFileSize()))
        .isEqualTo(digest1);
    assertThat(getFastDigestCounter.get()).isEqualTo(2);
    assertThat(getDigestCounter.get()).isEqualTo(1); // Cached.
  }

  @Test
  public void manuallyComputeDigest() throws Exception {
    byte[] digest = {1, 2, 3};
    FileSystem noDigestFileSystem =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          protected byte[] getFastDigest(PathFragment path) {
            throw new AssertionError("Unexpected call to getFastDigest");
          }

          @Override
          protected byte[] getDigest(PathFragment path) {
            return digest;
          }
        };
    Path file = noDigestFileSystem.getPath("/f.txt");
    FileSystemUtils.writeContentAsLatin1(file, "contents");

    assertThat(DigestUtils.manuallyComputeDigest(file, /*fileSize=*/ 8)).isEqualTo(digest);
  }
}
