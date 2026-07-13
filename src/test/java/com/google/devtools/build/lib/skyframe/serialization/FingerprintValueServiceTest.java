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
import static java.util.concurrent.Executors.newSingleThreadExecutor;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.SettableWriteStatus;
import java.util.concurrent.Executor;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FingerprintValueServiceTest {
  @Test
  public void fingerprint_isConsistent() {
    FingerprintValueService service =
        new FingerprintValueService(
            newSingleThreadExecutor(),
            new InMemoryFingerprintValueStore(),
            new FingerprintValueCache(),
            FingerprintValueService.NONPROD_FINGERPRINTER);

    assertThat(service.fingerprintPlaceholder().toBytes().length).isEqualTo(16);
    assertThat(service.fingerprintLength()).isEqualTo(16);

    byte[] testValue = new byte[] {0, 1, 2};
    PackedFingerprint testFingerprint = service.fingerprint(testValue);

    assertThat(testFingerprint).isNotEqualTo(service.fingerprintPlaceholder());
    assertThat(testFingerprint.toBytes().length).isEqualTo(16);
  }

  @Test
  public void put_delegatesToStore() {
    SettableWriteStatus expectedStatus = new SettableWriteStatus();
    FingerprintValueStore store =
        new FingerprintValueStore() {
          @Override
          public WriteStatus put(KeyBytesProvider fingerprint, byte[] serializedBytes) {
            return expectedStatus;
          }

          @Override
          public ListenableFuture<byte[]> get(KeyBytesProvider fingerprint) {
            throw new UnsupportedOperationException();
          }
        };

    FingerprintValueService service =
        new FingerprintValueService(
            newSingleThreadExecutor(),
            store,
            new FingerprintValueCache(),
            FingerprintValueService.NONPROD_FINGERPRINTER);

    byte[] testValue = new byte[] {0, 1, 2};
    PackedFingerprint testFingerprint = service.fingerprint(testValue);
    WriteStatus writeStatus = service.put(testFingerprint, testValue);

    assertThat(writeStatus).isSameInstanceAs(expectedStatus);
  }

  @Test
  public void executor_passesThrough() {
    Executor executor = newSingleThreadExecutor();
    FingerprintValueService service =
        new FingerprintValueService(
            executor,
            new InMemoryFingerprintValueStore(),
            new FingerprintValueCache(),
            FingerprintValueService.NONPROD_FINGERPRINTER);
    assertThat(service.getExecutor()).isSameInstanceAs(executor);
  }
}
