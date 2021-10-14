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
  public void putFutureIfAbsent_newFingerprint_returnsNull() {
    ByteString fingerprint1 = ByteString.copyFromUtf8("abc");
    ByteString fingerprint2 = ByteString.copyFromUtf8("xyz");
    SettableFuture<Object[]> future1 = SettableFuture.create();
    SettableFuture<Object[]> future2 = SettableFuture.create();

    assertThat(cache.putFutureIfAbsent(fingerprint1, future1)).isNull();
    assertThat(cache.putFutureIfAbsent(fingerprint2, future2)).isNull();
  }

  @Test
  public void putFutureIfAbsent_existingFingerprint_returnsExistingFuture() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future1 = SettableFuture.create();

    assertThat(cache.putFutureIfAbsent(fingerprint, future1)).isNull();
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create()))
        .isSameInstanceAs(future1);
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create()))
        .isSameInstanceAs(future1);
  }

  @Test
  public void putFutureIfAbsent_rejectsAlreadyDoneFuture() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();
    future.set(new Object[0]);

    assertThrows(
        IllegalArgumentException.class, () -> cache.putFutureIfAbsent(fingerprint, future));
  }

  @Test
  public void putFutureIfAbsent_futureCompletes_unwrapsContents() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future1 = SettableFuture.create();
    SettableFuture<Object[]> future2 = SettableFuture.create();
    Object[] contents = new Object[0];

    cache.putFutureIfAbsent(fingerprint, future1);
    future1.set(contents);

    assertThat(cache.putFutureIfAbsent(fingerprint, future2)).isSameInstanceAs(contents);
  }

  @Test
  public void putFutureIfAbsent_futureCompletes_cachesFingerprint() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();
    Object[] contents = new Object[0];

    cache.putFutureIfAbsent(fingerprint, future);
    future.set(contents);

    FingerprintComputationResult result = cache.fingerprintForContents(contents);
    assertThat(result.fingerprint()).isEqualTo(fingerprint);
    assertThat(result.writeStatus().isDone()).isTrue();
  }

  @Test
  public void putFutureIfAbsent_futureFails_notifiesBugReporter() {
    BugReporter mockBugReporter = mock(BugReporter.class);
    NestedSetSerializationCache cacheWithCustomBugReporter =
        new NestedSetSerializationCache(mockBugReporter);
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();
    Exception e = new MissingNestedSetException(fingerprint);

    cacheWithCustomBugReporter.putFutureIfAbsent(fingerprint, future);
    future.setException(e);

    verify(mockBugReporter).sendBugReport(e);
  }

  @Test
  public void putIfAbsent_newFingerprintAndContents_returnsNullAndCachesBothDirections() {
    ByteString fingerprint1 = ByteString.copyFromUtf8("abc");
    ByteString fingerprint2 = ByteString.copyFromUtf8("xyz");
    Object[] contents1 = new Object[] {"abc"};
    Object[] contents2 = new Object[] {"xyz"};
    FingerprintComputationResult result1 =
        FingerprintComputationResult.create(fingerprint1, SettableFuture.create());
    FingerprintComputationResult result2 =
        FingerprintComputationResult.create(fingerprint2, SettableFuture.create());

    assertThat(cache.putIfAbsent(contents1, result1)).isNull();
    assertThat(cache.putIfAbsent(contents2, result2)).isNull();
    assertThat(cache.fingerprintForContents(contents1)).isSameInstanceAs(result1);
    assertThat(cache.fingerprintForContents(contents2)).isSameInstanceAs(result2);
    assertThat(cache.putFutureIfAbsent(fingerprint1, SettableFuture.create()))
        .isSameInstanceAs(contents1);
    assertThat(cache.putFutureIfAbsent(fingerprint2, SettableFuture.create()))
        .isSameInstanceAs(contents2);
  }

  @Test
  public void putIfAbsent_existingFingerprintAndContents_returnsExistingResult() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    Object[] contents = new Object[0];
    FingerprintComputationResult result1 =
        FingerprintComputationResult.create(fingerprint, SettableFuture.create());
    FingerprintComputationResult result2 =
        FingerprintComputationResult.create(fingerprint, SettableFuture.create());
    FingerprintComputationResult result3 =
        FingerprintComputationResult.create(fingerprint, SettableFuture.create());

    assertThat(cache.putIfAbsent(contents, result1)).isNull();
    assertThat(cache.putIfAbsent(contents, result2)).isSameInstanceAs(result1);
    assertThat(cache.putIfAbsent(contents, result3)).isSameInstanceAs(result1);
  }

  @Test
  public void putIfAbsent_calledDuringPendingDeserialization_overwritesFuture() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();
    Object[] contents = new Object[0];
    FingerprintComputationResult result =
        FingerprintComputationResult.create(fingerprint, SettableFuture.create());

    assertThat(cache.putFutureIfAbsent(fingerprint, future)).isNull();
    assertThat(cache.putIfAbsent(contents, result)).isNull();
    assertThat(cache.fingerprintForContents(contents)).isSameInstanceAs(result);
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create()))
        .isSameInstanceAs(contents);

    // After the future completes, the contents should still be cached (doesn't matter which array).
    Object[] deserializedContents = new Object[0];
    future.set(deserializedContents);
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create()))
        .isAnyOf(contents, deserializedContents);

    // Both arrays should have a FingerprintComputationResult.
    FingerprintComputationResult resultForDeserializedContents =
        cache.fingerprintForContents(deserializedContents);
    assertThat(resultForDeserializedContents.fingerprint()).isEqualTo(fingerprint);
    assertThat(resultForDeserializedContents.writeStatus().isDone()).isTrue();
    assertThat(cache.fingerprintForContents(contents)).isSameInstanceAs(result);
  }

  @Test
  public void putFutureIfAbsent_cacheEntriesHaveLifetimeOfContents() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    SettableFuture<Object[]> future = SettableFuture.create();

    cache.putFutureIfAbsent(fingerprint, future);

    // Before completing, still cached while future in memory.
    GcFinalization.awaitFullGc();
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create()))
        .isSameInstanceAs(future);

    // After completing, still cached while contents in memory even if future is gone.
    WeakReference<SettableFuture<Object[]>> futureRef = new WeakReference<>(future);
    Object[] contents = new Object[0];
    future.set(contents);
    future = null;
    GcFinalization.awaitClear(futureRef);
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create()))
        .isSameInstanceAs(contents);
    FingerprintComputationResult result = cache.fingerprintForContents(contents);
    assertThat(result.fingerprint()).isEqualTo(fingerprint);
    assertThat(result.writeStatus().isDone()).isTrue();

    // Cleared after references are gone, and the cycle of putFutureIfAbsent starts over.
    WeakReference<Object[]> contentsRef = new WeakReference<>(contents);
    contents = null;
    GcFinalization.awaitClear(contentsRef);
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create())).isNull();
  }

  @Test
  public void putIfAbsent_cacheEntriesHaveLifetimeOfContents() {
    ByteString fingerprint = ByteString.copyFromUtf8("abc");
    Object[] contents = new Object[0];
    FingerprintComputationResult result =
        FingerprintComputationResult.create(fingerprint, immediateVoidFuture());

    cache.putIfAbsent(contents, result);

    // Still cached while in memory.
    GcFinalization.awaitFullGc();
    assertThat(cache.fingerprintForContents(contents)).isSameInstanceAs(result);
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create()))
        .isSameInstanceAs(contents);

    // Cleared after references are gone, and the cycle of putFutureIfAbsent starts over.
    WeakReference<Object[]> ref = new WeakReference<>(contents);
    contents = null;
    GcFinalization.awaitClear(ref);
    assertThat(cache.putFutureIfAbsent(fingerprint, SettableFuture.create())).isNull();
  }
}
