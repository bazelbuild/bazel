// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.util.Fingerprint;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DigestMap}. */
@RunWith(JUnit4.class)
public class DigestMapTest {

  @Test
  public void simpleTest() {
    int count = 128; // Must be smaller than byte for this test or we'll overflow
    Object[] keys = new Object[count];
    for (int i = 0; i < count; ++i) {
      keys[i] = new Object();
    }

    DigestMap digestMap = new DigestMap(16, 4);
    for (int i = 0; i < count; ++i) {
      Fingerprint digest = new Fingerprint().addInt(i);
      Fingerprint fingerprint = new Fingerprint();
      digestMap.insertAndReadDigest(keys[i], digest, fingerprint);
      Fingerprint reference =
          new Fingerprint().addBytes(new Fingerprint().addInt(i).digestAndReset());
      assertThat(fingerprint.hexDigestAndReset()).isEqualTo(reference.hexDigestAndReset());
    }
    for (int i = 0; i < count; ++i) {
      Fingerprint fingerprint = new Fingerprint();
      assertThat(digestMap.readDigest(keys[i], fingerprint)).isTrue();
      Fingerprint reference =
          new Fingerprint().addBytes(new Fingerprint().addInt(i).digestAndReset());
      assertThat(fingerprint.hexDigestAndReset()).isEqualTo(reference.hexDigestAndReset());
    }
  }

  @Test
  public void concurrencyTest() throws Exception {
    int count = 128; // Must be smaller than byte for this test or we'll overflow
    Object[] keys = new Object[count];
    for (int i = 0; i < count; ++i) {
      keys[i] = new Object();
    }
    DigestMap digestMap = new DigestMap(16, 4);

    AtomicBoolean done = new AtomicBoolean();
    AtomicReference<Exception> exception = new AtomicReference<>();
    List<Thread> threads = new ArrayList<>();
    int threadCount = 16;
    for (int i = 0; i < threadCount; ++i) {
      Thread thread =
          new Thread(
              () -> {
                Random random = new Random();
                while (!done.get()) {
                  int index = random.nextInt(count);
                  Object key = keys[index];
                  Fingerprint fingerprint = new Fingerprint();
                  if (!digestMap.readDigest(key, fingerprint)) {
                    Fingerprint digest = new Fingerprint().addInt(index);
                    digestMap.insertAndReadDigest(key, digest, fingerprint);
                  }
                  Fingerprint reference =
                      new Fingerprint().addBytes(new Fingerprint().addInt(index).digestAndReset());
                  String hexDigest = fingerprint.hexDigestAndReset();
                  String referenceDigest = reference.hexDigestAndReset();
                  if (!hexDigest.equals(referenceDigest)) {
                    exception.set(
                        new IllegalStateException(
                            String.format(
                                "Digests are not equal: %s != %s, index %d",
                                hexDigest, referenceDigest, index)));
                    done.set(true);
                  }
                }
              });
      thread.start();
      threads.add(thread);
    }
    Thread.sleep(1000);
    done.set(true);
    for (int i = 0; i < threadCount; ++i) {
      threads.get(i).join(1000);
    }
    if (exception.get() != null) {
      throw exception.get();
    }
  }
}
