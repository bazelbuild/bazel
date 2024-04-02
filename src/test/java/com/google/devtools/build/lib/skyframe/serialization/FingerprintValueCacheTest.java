// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class FingerprintValueCacheTest {
  // FingerprintValueService delegates getOrClaimPutOperation and getOrClaimGetOperation to
  // FingerprintValueCache. The tests here test the logic in FingerprintValueCache but go through
  // through the FingerprintValueService to improve coverage there.

  enum Distinguisher {
    NULL_DISTINGUISHER(/* value= */ null),
    NON_NULL_DISTINGUISHER(/* value= */ new Object());

    @SuppressWarnings("ImmutableEnumChecker") // always an immutable instance
    @Nullable
    private final Object value;

    Distinguisher(@Nullable Object value) {
      this.value = value;
    }

    @Nullable
    Object value() {
      return value;
    }
  }

  @Test
  public void putOperation_isCached(@TestParameter Distinguisher distinguisher) {
    FingerprintValueService service =
        FingerprintValueService.createForTesting(/* exerciseDeserializationForTesting= */ false);

    Object value = new Object();

    SettableFuture<PutOperation> op1 = SettableFuture.create();
    Object result = service.getOrClaimPutOperation(value, distinguisher.value(), op1);
    assertThat(result).isNull();

    SettableFuture<PutOperation> op2 = SettableFuture.create();
    result = service.getOrClaimPutOperation(value, distinguisher.value(), op2);
    assertThat(result).isSameInstanceAs(op1);
  }

  @Test
  public void getOperation_isCached(@TestParameter Distinguisher distinguisher) {
    FingerprintValueService service =
        FingerprintValueService.createForTesting(/* exerciseDeserializationForTesting= */ false);

    ByteString fingerprint = ByteString.copyFromUtf8("foo");

    SettableFuture<Object> op1 = SettableFuture.create();
    Object result = service.getOrClaimGetOperation(fingerprint, distinguisher.value(), op1);
    assertThat(result).isNull();

    SettableFuture<Object> op2 = SettableFuture.create();
    result = service.getOrClaimGetOperation(fingerprint, distinguisher.value(), op2);
    assertThat(result).isSameInstanceAs(op1);
  }

  @Test
  public void putOperation_isUnwrapped(@TestParameter Distinguisher distinguisher) {
    FingerprintValueService service =
        FingerprintValueService.createForTesting(/* exerciseDeserializationForTesting= */ false);

    Object value = new Object();

    SettableFuture<PutOperation> putOp1 = SettableFuture.create();
    Object putResult = service.getOrClaimPutOperation(value, distinguisher.value(), putOp1);
    assertThat(putResult).isNull();

    // Sets the `PutOperation` in `putOp1`, which triggers the first stage of unwrapping and
    // populates the reverse service.
    ByteString fingerprint = ByteString.copyFromUtf8("foo");
    SettableFuture<Void> writeStatus = SettableFuture.create();
    putOp1.set(PutOperation.create(fingerprint, writeStatus));

    // A get of `fingerprint` now returns `value` immediately.
    SettableFuture<Object> getOp = SettableFuture.create();
    Object getResult = service.getOrClaimGetOperation(fingerprint, distinguisher.value(), getOp);
    assertThat(getResult).isSameInstanceAs(value);

    // A second "put" of `value' sees the original, wrapped `putOp1`.
    SettableFuture<PutOperation> putOp2 = SettableFuture.create();
    putResult = service.getOrClaimPutOperation(value, distinguisher.value(), putOp2);
    assertThat(putResult).isSameInstanceAs(putOp1);

    // Setting the write status fully unwraps the value.
    writeStatus.set(null);
    putResult = service.getOrClaimPutOperation(value, distinguisher.value(), putOp2);
    assertThat(putResult).isSameInstanceAs(fingerprint);
  }

  @Test
  public void getOperation_isUnwrapped(@TestParameter Distinguisher distinguisher) {
    FingerprintValueService service =
        FingerprintValueService.createForTesting(/* exerciseDeserializationForTesting= */ false);

    ByteString fingerprint = ByteString.copyFromUtf8("foo");

    SettableFuture<Object> getOp = SettableFuture.create();
    Object result = service.getOrClaimGetOperation(fingerprint, distinguisher.value(), getOp);
    assertThat(result).isNull();

    // The first put operation is owned by the caller.
    Object value = new Object();
    SettableFuture<PutOperation> putOp = SettableFuture.create();
    Object putResult = service.getOrClaimPutOperation(value, distinguisher.value(), putOp);
    assertThat(putResult).isNull();

    // Completes the `getOp`, causing it to be unwrapped.
    getOp.set(value);

    // The next put operation gets the unwrapped fingerprint.
    SettableFuture<PutOperation> putOp2 = SettableFuture.create();
    putResult = service.getOrClaimPutOperation(value, distinguisher.value(), putOp2);
    assertThat(putResult).isSameInstanceAs(fingerprint);

    // Completing `putOp` overwrites values, but this is benign because `value`s fingerprint should
    // be deterministic.
    ByteString fingerprint2 = ByteString.copyFromUtf8("foo");
    putOp.set(PutOperation.create(fingerprint2, immediateVoidFuture()));

    SettableFuture<PutOperation> putOp3 = SettableFuture.create();
    putResult = service.getOrClaimPutOperation(value, distinguisher.value(), putOp3);
    assertThat(putResult).isSameInstanceAs(fingerprint2);
  }

  @Test
  public void distinguisher_distinguishesSameFingerprint() {
    // Puts two values with the same fingerprint, but different distinguishers, then verifies that
    // they are distinguishable on retrieval.
    FingerprintValueService service =
        FingerprintValueService.createForTesting(/* exerciseDeserializationForTesting= */ false);

    ByteString fingerprint = ByteString.copyFromUtf8("foo");

    ListenableFuture<PutOperation> put =
        immediateFuture(PutOperation.create(fingerprint, immediateVoidFuture()));

    Object value1 = new Object();
    Object distinguisher1 = new Object();
    Object result = service.getOrClaimPutOperation(value1, distinguisher1, put);
    assertThat(result).isNull();

    Object value2 = new Object();
    Object distinguisher2 = new Object();
    // Reusing `put` here is fine because it's the same fingerprint.
    result = service.getOrClaimPutOperation(value2, distinguisher2, put);
    assertThat(result).isNull();

    // The correct values are returned for the distinguisher values.
    SettableFuture<Object> unusedGetOperation = SettableFuture.create();
    result = service.getOrClaimGetOperation(fingerprint, distinguisher1, unusedGetOperation);
    assertThat(result).isSameInstanceAs(value1);
    result = service.getOrClaimGetOperation(fingerprint, distinguisher2, unusedGetOperation);
    assertThat(result).isSameInstanceAs(value2);
  }

  @Test
  public void exerciseDeserializationForTesting_doesNotAddReverseEntry(
      @TestParameter Distinguisher distinguisher,
      @TestParameter boolean exerciseDeserializationForTesting) {
    FingerprintValueService service =
        FingerprintValueService.createForTesting(exerciseDeserializationForTesting);

    // Puts the `fingerprint` to `value` association in the service.
    ByteString fingerprint = ByteString.copyFromUtf8("foo");
    Object value = new Object();
    Object result =
        service.getOrClaimPutOperation(
            value,
            distinguisher.value(),
            immediateFuture(PutOperation.create(fingerprint, immediateVoidFuture())));
    assertThat(result).isNull();

    SettableFuture<Object> getOperation = SettableFuture.create();
    result = service.getOrClaimGetOperation(fingerprint, distinguisher.value(), getOperation);
    if (exerciseDeserializationForTesting) {
      // Ordinarily, the reverse `value` to `fingerprint` would also be added to the cache, but it's
      // not because `exerciseDeserializationForTesting` is true.
      assertThat(result).isNull();
    } else {
      assertThat(result).isSameInstanceAs(value);
    }
  }
}
