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

import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import com.google.protobuf.ByteString;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.concurrent.Executor;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class FingerprintValueServiceTest {
  // FingerprintValueService is a thin wrapper over a few loosely related objects. The contained
  // FingerprintValueCache is covered in FingerprintValueCacheTest.

  @Test
  public void get_returnsPreviouslyPut() throws Exception {
    // `get` and `put` delegate to an underlying InMemoryFingerprintValueStore. Sanity checks this
    // wiring.

    FingerprintValueService service = FingerprintValueService.createForTesting();
    ByteString key = ByteString.copyFromUtf8("key");
    byte[] value = new byte[] {0, 1, 2};

    Void unused = service.put(key, value).get();

    assertThat(service.get(key).get()).isSameInstanceAs(value);
  }

  enum FingerprintFunction {
    MURMUR3_64(Hashing.murmur3_32_fixed()),
    MURMUR3_128(Hashing.murmur3_128()),
    SHA_256(Hashing.sha256());

    private final HashFunction hashFunction;

    FingerprintFunction(HashFunction hashFunction) {
      this.hashFunction = hashFunction;
    }

    HashFunction hashFunction() {
      return hashFunction;
    }

    int expectedLength() {
      return hashFunction.bits() / 8;
    }
  }

  @Test
  public void fingerprintConsistent(@TestParameter FingerprintFunction fingerprinter) {
    FingerprintValueService service =
        new FingerprintValueService(
            newSingleThreadExecutor(),
            FingerprintValueStore.inMemoryStore(),
            new FingerprintValueCache(),
            fingerprinter.hashFunction());

    assertThat(service.fingerprintPlaceholder().size()).isEqualTo(fingerprinter.expectedLength());
    assertThat(service.fingerprintLength()).isEqualTo(fingerprinter.expectedLength());

    byte[] testValue = new byte[] {0, 1, 2};
    ByteString testFingerprint = service.fingerprint(testValue);

    assertThat(testFingerprint).isNotEqualTo(service.fingerprintPlaceholder());
    assertThat(testFingerprint.size()).isEqualTo(fingerprinter.expectedLength());
  }

  @Test
  public void exectutor_passesThrough() {
    Executor executor = newSingleThreadExecutor();
    FingerprintValueService service =
        new FingerprintValueService(
            executor,
            FingerprintValueStore.inMemoryStore(),
            new FingerprintValueCache(),
            Hashing.murmur3_128());
    assertThat(service.getExecutor()).isSameInstanceAs(executor);
  }
}
