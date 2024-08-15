// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Basic tests for {@link PackageIdentifier}'s codec. */
@RunWith(JUnit4.class)
public class PackageIdentifierCodecTest {

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(PackageIdentifier.create("foo", PathFragment.create("bar/baz")))
        .runTests();
  }

  @Test
  public void sharedValueCodec_works() throws Exception {
    new SerializationTester(PackageIdentifier.create("foo", PathFragment.create("bar/baz")))
        .addCodec(PackageIdentifier.valueSharingCodec())
        .makeMemoizingAndAllowFutureBlocking(true)
        .runTests();
  }
}
