// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.singlejar;

import static com.google.common.truth.Truth.assertThat;

import java.io.IOException;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link CopyEntryFilter}. */
@RunWith(JUnit4.class)
public class CopyEntryFilterTest {

  @Test
  public void testSingleInput() throws IOException {
    RecordingCallback callback = new RecordingCallback();
    new CopyEntryFilter().accept("abc", callback);
    assertThat(callback.calls).isEqualTo(Arrays.asList("copy"));
  }

}
