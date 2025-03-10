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

import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import java.util.concurrent.locks.ReentrantLock;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ReentrantLockCodecTest {

  @Test
  public void codec_roundTrips() throws Exception {
    var subject = new ReentrantLock();
    ReentrantLock deserialized = RoundTripping.roundTrip(subject, AutoRegistry.get());
    assertThat(deserialized).isNotNull();
    assertThat(deserialized.isLocked()).isFalse();
  }

  @Test
  public void roundTrip_unlocks() throws Exception {
    var subject = new ReentrantLock();
    subject.lock();
    assertThat(subject.isLocked()).isTrue();
    ReentrantLock deserialized = RoundTripping.roundTrip(subject, AutoRegistry.get());
    assertThat(deserialized).isNotNull();
    assertThat(deserialized.isLocked()).isFalse();
  }
}
