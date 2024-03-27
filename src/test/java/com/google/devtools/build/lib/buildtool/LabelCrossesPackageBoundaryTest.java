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

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration test for labels that cross package boundaries.
 */
@RunWith(JUnit4.class)
public class LabelCrossesPackageBoundaryTest extends BuildIntegrationTestCase {

  @Test
  public void testLabelCrossesPackageBoundary_target() throws Exception {
    write(
        "x/BUILD",
        """
        genrule(
            name = "x",
            srcs = ["//x:y/z"],
            outs = ["out"],
            cmd = "true",
        )
        """);
    write("x/y/BUILD",
          "exports_files(['z'])");

    assertThrows(TargetParsingException.class, () -> buildTarget("//x"));

    events.assertContainsError("Label '//x:y/z' is invalid because 'x/y' is a subpackage");
  }
}
