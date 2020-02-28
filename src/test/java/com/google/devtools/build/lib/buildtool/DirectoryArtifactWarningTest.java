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
package com.google.devtools.build.lib.buildtool;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.packages.util.MockGenruleSupport;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration test for warnings issued when an artifact is a directory.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class DirectoryArtifactWarningTest extends BuildIntegrationTestCase {

  @Test
  public void testOutputArtifactDirectoryWarning() throws Exception {
    MockGenruleSupport.setup(mockToolsConfig);
    write("x/BUILD",
          "genrule(name = 'x',",
          "        outs = ['dir'],",
          "        cmd = '/bin/mkdir $(location dir)',",
          "        srcs = [])");

    buildTarget("//x");

    events.assertContainsWarning("output 'x/dir' of //x:x is a directory; "
                        + "dependency checking of directories is unsound");
  }

  @Test
  public void testInputArtifactDirectoryWarning() throws Exception {
    MockGenruleSupport.setup(mockToolsConfig);
    write("x/BUILD",
          "genrule(name = 'x',",
          "        outs = ['out'],",
          "        cmd = '/bin/touch $(location out)',",
          "        srcs = ['dir'])");
    write("x/dir/empty");

    buildTarget("//x");

    events.assertContainsWarning("input 'x/dir' to //x:x is a directory; "
          + "dependency checking of directories is unsound");
  }

}
