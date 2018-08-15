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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/** Tests for {@link DigestMap}. */
@RunWith(Parameterized.class)
public class DigestMapTest {

  @Parameters(name = "Hash: {0}")
  public static Iterable<Object[]> hashFunction() {
    return ImmutableList.of(
        new Object[] {DigestHashFunction.MD5}, new Object[] {DigestHashFunction.SHA256});
  }

  @Parameter public DigestHashFunction digestHashFunction;

  private Fingerprint fingerprint() {
    return new Fingerprint(digestHashFunction);
  }

  @Test
  public void simpleTest() {
    int count = 128; // Must be smaller than byte for this test or we'll overflow
    Object[] keys = new Object[count];
    for (int i = 0; i < count; ++i) {
      keys[i] = new Object();
    }

    DigestMap digestMap = new DigestMap(digestHashFunction, 4);
    for (int i = 0; i < count; ++i) {
      Fingerprint digest = fingerprint().addInt(i);
      Fingerprint fingerprint = fingerprint();
      digestMap.insertAndReadDigest(keys[i], digest, fingerprint);
      Fingerprint reference = fingerprint().addBytes(fingerprint().addInt(i).digestAndReset());
      assertThat(fingerprint.hexDigestAndReset()).isEqualTo(reference.hexDigestAndReset());
    }
    for (int i = 0; i < count; ++i) {
      Fingerprint fingerprint = fingerprint();
      assertThat(digestMap.readDigest(keys[i], fingerprint)).isTrue();
      Fingerprint reference = fingerprint().addBytes(fingerprint().addInt(i).digestAndReset());
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
    DigestMap digestMap = new DigestMap(digestHashFunction, 4);

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
                  Fingerprint fingerprint = fingerprint();
                  if (!digestMap.readDigest(key, fingerprint)) {
                    Fingerprint digest = fingerprint().addInt(index);
                    digestMap.insertAndReadDigest(key, digest, fingerprint);
                  }
                  Fingerprint reference =
                      fingerprint().addBytes(fingerprint().addInt(index).digestAndReset());
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
