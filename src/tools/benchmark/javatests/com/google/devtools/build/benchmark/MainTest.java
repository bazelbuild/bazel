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

package com.google.devtools.build.benchmark;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link Main}. */
@RunWith(JUnit4.class)
public class MainTest {

  @Test
  public void testParseArgsMissingArgs() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[] {"--workspace=workspace", "--from=1"});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertEquals("Argument value should not be empty.", e.getMessage());
    }
  }

  @Test
  public void testParseArgsCorrect() throws OptionsParsingException, IOException {
    BenchmarkOptions opt =
        Main.parseArgs(
            new String[] {"--output=output", "--workspace=workspace", "--from=1", "--to=3"});
    assertEquals(opt.output, "output");
    assertEquals(opt.workspace, "workspace");
    assertEquals(opt.from, "1");
    assertEquals(opt.to, "3");
  }
}
