// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class LocationTest {

  @Test
  public void fromFile() throws Exception {
    String file = "this is a filename";
    Location location = Location.fromFile(file);
    assertThat(location.file()).isEqualTo(file);
    assertThat(location.line()).isEqualTo(0);
    assertThat(location.column()).isEqualTo(0);
    assertThat(location.toString()).isEqualTo(file);
  }

  @Test
  public void testCodec() throws Exception {
    String file = "this is a filename";
    new SerializationTester(
            Location.fromFile(file), //
            Location.fromFileLineColumn(file, 20, 25),
            Location.BUILTIN)
        .runTests();
  }
}
