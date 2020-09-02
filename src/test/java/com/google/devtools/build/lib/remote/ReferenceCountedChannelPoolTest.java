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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.remote.ReferenceCountedChannelPool} */
@RunWith(JUnit4.class)
public class ReferenceCountedChannelPoolTest {
  @Test
  public void getChannelIndexTest() {
    int poolSize = Integer.MAX_VALUE;
    int affinity = Integer.MIN_VALUE;
    int index = ReferenceCountedChannelPool.getChannelIndex(poolSize, affinity);
    assertThat(index >= 0).isTrue();
    assertThat(index < poolSize).isTrue();
  }
}
