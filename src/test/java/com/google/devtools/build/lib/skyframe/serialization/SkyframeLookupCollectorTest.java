// Copyright 2025 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.PeerFailedException;
import com.google.devtools.build.lib.skyframe.serialization.SharedValueDeserializationContext.SkyframeLookup;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SkyframeLookupCollectorTest {
  private static final SkyFunctionName DUMMY_NAME = SkyFunctionName.createHermetic("FOR_TESTING");

  private final SkyframeLookupCollector collector = new SkyframeLookupCollector();

  @Test
  public void exceptionNotification_marksAllLookupsAbandoned() {
    var parent1 = new AtomicReference<Object>();
    var lookup1 =
        new SkyframeLookup<AtomicReference<Object>>(
            createDummyKey(), parent1, AtomicReference::set);

    var parent2 = new AtomicReference<Object>();
    var lookup2 =
        new SkyframeLookup<AtomicReference<Object>>(
            createDummyKey(), parent2, AtomicReference::set);

    collector.addLookup(lookup1);
    collector.addLookup(lookup2);

    var exception = new Exception("failed");
    collector.notifyFetchException(exception);

    var thrown = assertThrows(ExecutionException.class, collector::get);
    assertThat(thrown).hasCauseThat().isSameInstanceAs(exception);

    // Verifies that enqueued lookups are abandoned.
    assertHasPeerFailure(lookup1, exception);
    assertHasPeerFailure(lookup2, exception);

    // Verifies that subsequently added lookups are abandoned.
    var parent3 = new AtomicReference<Object>();
    var lookup3 =
        new SkyframeLookup<AtomicReference<Object>>(
            createDummyKey(), parent3, AtomicReference::set);
    collector.addLookup(lookup3);
    assertHasPeerFailure(lookup3, exception);
  }

  private static void assertHasPeerFailure(SkyframeLookup<?> lookup, Exception exception) {
    assertThat(lookup.isFailed()).isTrue();
    var thrown = assertThrows(ExecutionException.class, lookup::get);
    var cause = thrown.getCause();
    assertThat(cause).isInstanceOf(PeerFailedException.class);
    assertThat(cause).hasCauseThat().isSameInstanceAs(exception);
  }

  private static SkyKey createDummyKey() {
    return () -> DUMMY_NAME;
  }
}
