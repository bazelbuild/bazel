// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SingleStatRecorder}. */
@RunWith(JUnit4.class)
public class SingleStatRecorderTest {

  @Test
  public void meanAndstandardDeviation() {
    SingleStatRecorder recorder = new SingleStatRecorder(null, 2);
    recorder.addStat(43, null);
    recorder.addStat(49, null);
    recorder.addStat(51, null);
    recorder.addStat(57, null);
    MetricData metrics = recorder.snapshot();
    assertThat(metrics.getAvg()).isWithin(.01).of(50);
    assertThat(metrics.getStdDev()).isWithin(.01).of(5);
  }
}
