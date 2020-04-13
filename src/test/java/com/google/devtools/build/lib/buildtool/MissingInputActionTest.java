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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.buildtool.util.GoogleBuildIntegrationTestCase;
import com.google.devtools.build.lib.packages.util.MockGenruleSupport;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests related to "missing input file" errors.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class MissingInputActionTest extends GoogleBuildIntegrationTestCase {
  protected String prefix = "//";
  protected String separator = ":";

  // Regression test for bug #904676: Blaze does not consider missing inputs
  // an error.
  // TODO(nsakharo) make it purely lib.actions test instead of buildtool test.
  @Test
  public void testNoInput() throws Exception {
    // Multiple missing inputs means error is non-deterministic in --nokeep_going case.
    this.addOptions("--keep_going");
    MockGenruleSupport.setup(mockToolsConfig);
    write("dummy/BUILD",
          "genrule(name = 'dummy', ",
          "        srcs = ['in1', 'in2', 'in3'], ",
          "        outs = ['out1', 'out2'],  ",
          "        cmd = '/bin/true')");
    write("dummy/in1");

    assertMissingInputOnBuild("//dummy", /*don't check error message*/0);
    events.assertDoesNotContainEvent(
        "missing input file '" + prefix + "dummy" + separator + "in1'");
    events.assertContainsError("missing input file '" + prefix + "dummy" + separator + "in2'");
    events.assertContainsError("missing input file '" + prefix + "dummy" + separator + "in3'");
  }

  private void assertMissingInputOnBuild(String label, int num_missing) throws Exception {
    BuildFailedException e = assertThrows(BuildFailedException.class, () -> buildTarget(label));
    if (num_missing > 0) {
        String expected = num_missing + " input file(s) do not exist";
        assertThat(e).hasMessageThat().contains(expected);
        assertWithMessage("Culprit action should not be null: " + e)
            .that(e.getAction())
            .isNotNull();
    }
  }
}
