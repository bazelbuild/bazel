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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfTargetsUnderDirectoryValue.PrepareDepsOfTargetsUnderDirectoryKey;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PrepareDepsOfTargetsUnderDirectoryKey}'s codec. */
@RunWith(JUnit4.class)
public final class PrepareDepsOfTargetsUnderDirectoryKeyCodecTest {

  @Test
  public void testCodec() throws Exception {
    SerializationTester serializationTester =
        new SerializationTester(
            PrepareDepsOfTargetsUnderDirectoryKey.create(
                new RecursivePkgKey(
                    RepositoryName.MAIN,
                    FsUtils.TEST_ROOTED_PATH,
                    ImmutableSet.of(FsUtils.rootPathRelative("here"))),
                FilteringPolicies.and(
                    FilteringPolicies.NO_FILTER, FilteringPolicies.FILTER_TESTS)));
    FsUtils.addDependencies(serializationTester);
    serializationTester.runTests();
  }
}
