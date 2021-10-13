// Copyright 2021 The Bazel Authors. All rights reserved.
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
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.google.common.testing.GcFinalization;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.FingerprintComputationResult;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.MissingNestedSetException;
import com.google.protobuf.ByteString;
import java.lang.ref.WeakReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NestedSetSerializationCache}. */
@RunWith(JUnit4.class)
public final class NestedSetSerializationCacheTest {

  private final NestedSetSerializationCache cache =
      new NestedSetSerializationCache(BugReporter.defaultInstance());

  @Test
  public void putIfAbsent_newFingerprint_returnsNull() {
    ByteString fingerprint1 = ByteString.copyFromUtf8("abc");
    ByteString fingerprint2 = ByteString.copyFromUtf8("xyz");
    SettableFuture<Object[]> future1 = SettableFuture.create();
    SettableFuture<Object[]> future2 = SettableFuture.create();

    assertThat(cache.putIfAbsent(fingerprint1, future1)).isNull();
    assertThat(cache.putIfAbsent(fingerprint2, future2)).isNull();
  }

  @Test
  public void putIfAbsent_existingFingerprint_returnsExistingFuture() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future1 = SettableFuture.create();

    assertThat(cache.putIfAbsent(fingerprint, future1)).isNull();
    assertThat(cache.putIfAbsent(fingerprint, SettableFuture.create())).isSameInstanceAs(future1);
    assertThat(cache.putIfAbsent(fingerprint, SettableFuture.create())).isSameInstanceAs(future1);
  }

  @Test
  public void putIfAbsent_rejectsAlreadyDoneFuture() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();
    future.set(new Object[0]);

    assertThrows(IllegalArgumentException.class, () -> cache.putIfAbsent(fingerprint, future));
  }

  @Test
  public void putIfAbsent_futureCompletes_unwrapsContents() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future1 = SettableFuture.create();
    SettableFuture<Object[]> future2 = SettableFuture.create();
    Object[] contents = new Object[0];

    cache.putIfAbsent(fingerprint, future1);
    future1.set(contents);

    assertThat(cache.putIfAbsent(fingerprint, future2)).isSameInstanceAs(contents);
  }

  @Test
  public void putIfAbsent_futureCompletes_cachesFingerprint() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();
    Object[] contents = new Object[0];

    cache.putIfAbsent(fingerprint, future);
    future.set(contents);

    FingerprintComputationResult result = cache.fingerprintForContents(contents);
    assertThat(result.fingerprint()).isEqualTo(fingerprint);
    assertThat(result.writeStatus().isDone()).isTrue();
  }

  @Test
  public void putIfAbsent_futureFails_notifiesBugReporter() {
    BugReporter mockBugReporter = mock(BugReporter.class);
    NestedSetSerializationCache cacheWithCustomBugReporter =
        new NestedSetSerializationCache(mockBugReporter);
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();
    Exception e = new MissingNestedSetException(fingerprint);

    cacheWithCustomBugReporter.putIfAbsent(fingerprint, future);
    future.setException(e);

    verify(mockBugReporter).sendBugReport(e);
  }

  @Test
  public void put_cachesBothDirections() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    Object[] contents = new Object[0];
    FingerprintComputationResult result =
        FingerprintComputationResult.create(fingerprint, immediateVoidFuture());

    cache.put(result, contents);

    assertThat(cache.fingerprintForContents(contents)).isEqualTo(result);
    assertThat(cache.putIfAbsent(fingerprint, SettableFuture.create())).isSameInstanceAs(contents);
  }

  @Test
  public void putIfAbsent_cacheEntriesHaveLifetimeOfContents() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();

    cache.putIfAbsent(fingerprint, future);

    // Before completing, still cached while future in memory.
    GcFinalization.awaitFullGc();
    assertThat(cache.putIfAbsent(fingerprint, SettableFuture.create())).isSameInstanceAs(future);

    // After completing, still cached while contents in memory even if future is gone.
    WeakReference<SettableFuture<Object[]>> futureRef = new WeakReference<>(future);
    Object[] contents = new Object[0];
    future.set(contents);
    future = null;
    GcFinalization.awaitClear(futureRef);
    assertThat(cache.putIfAbsent(fingerprint, SettableFuture.create())).isSameInstanceAs(contents);
    FingerprintComputationResult result = cache.fingerprintForContents(contents);
    assertThat(result.fingerprint()).isEqualTo(fingerprint);
    assertThat(result.writeStatus().isDone()).isTrue();

    // Cleared after references are gone, and the cycle of putIfAbsent starts over.
    WeakReference<Object[]> contentsRef = new WeakReference<>(contents);
    contents = null;
    GcFinalization.awaitClear(contentsRef);
    assertThat(cache.putIfAbsent(fingerprint, SettableFuture.create())).isNull();
  }

  @Test
  public void put_cacheEntriesHaveLifetimeOfContents() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    Object[] contents = new Object[0];
    FingerprintComputationResult result =
        FingerprintComputationResult.create(fingerprint, immediateVoidFuture());

    cache.put(result, contents);

    // Still cached while in memory.
    GcFinalization.awaitFullGc();
    assertThat(cache.fingerprintForContents(contents)).isEqualTo(result);
    assertThat(cache.putIfAbsent(fingerprint, SettableFuture.create())).isSameInstanceAs(contents);

    // Cleared after references are gone, and the cycle of putIfAbsent starts over.
    WeakReference<Object[]> ref = new WeakReference<>(contents);
    contents = null;
    GcFinalization.awaitClear(ref);
    assertThat(cache.putIfAbsent(fingerprint, SettableFuture.create())).isNull();
  }
}
