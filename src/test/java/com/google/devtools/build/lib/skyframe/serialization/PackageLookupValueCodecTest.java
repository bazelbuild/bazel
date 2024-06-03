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

import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PackageLookupValueCodec}. */
@RunWith(JUnit4.class)
public class PackageLookupValueCodecTest {

  @Test
  public void testCodec() throws Exception {
    SerializationTester serializationTester =
        new SerializationTester(
            PackageLookupValue.success(
                Root.fromPath(FsUtils.TEST_FILESYSTEM.getPath("/success")), BuildFileName.BUILD),
            PackageLookupValue.success(
                Root.fromPath(FsUtils.TEST_FILESYSTEM.getPath("/success")),
                BuildFileName.WORKSPACE),
            PackageLookupValue.invalidPackageName("junkjunkjunk"),
            PackageLookupValue.NO_BUILD_FILE_VALUE,
            PackageLookupValue.DELETED_PACKAGE_VALUE);
    FsUtils.addDependencies(serializationTester);
    serializationTester.runTests();
  }
}
