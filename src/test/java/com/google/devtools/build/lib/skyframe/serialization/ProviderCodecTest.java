// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link BuiltinProvider} (de)serialization. */
@RunWith(JUnit4.class)
public final class ProviderCodecTest {
  private static class DummyProvider extends BuiltinProvider<StructImpl> {
    DummyProvider() {
      super("DummyInfo", StructImpl.class);
    }
  }

  @Test
  public void objectCodecTests() throws Exception {
    DummyProvider dummyProvider = new DummyProvider();
    new SerializationTester(dummyProvider)
        .addDependency(DummyProvider.class, dummyProvider)
        .runTests();
  }
}
