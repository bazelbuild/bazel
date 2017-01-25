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

package com.google.devtools.build.benchmark.codegenerator;

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
  public void testParseArgsEmpty() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[]{});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertEquals("--output_dir should not be empty.", e.getMessage());
    }
  }

  @Test
  public void testParseArgsWrongMode() throws IOException {
    try {
      Main.parseArgs(new String[]{"--modify=mango"});
      fail("Should throw OptionsParsingException");
    } catch (OptionsParsingException e) {
      assertEquals(
          "While parsing option --modify=mango: 'mango' is not a boolean", e.getMessage());
    }
  }

  @Test
  public void testParseArgsNoOutputDir() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[]{"--modify"});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertEquals("--output_dir should not be empty.", e.getMessage());
    }
  }

  @Test
  public void testParseArgsOutputDirNonExists() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[]{"--modify", "--output_dir=mango"});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertEquals("--output_dir (mango) does not contain code for modification.", e.getMessage());
    }
  }

  @Test
  public void testParseArgsNoType() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[]{"--output_dir=mango"});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertEquals("No type of package is specified.", e.getMessage());
    }
  }

  @Test
  public void testParseArgsCorrect() throws OptionsParsingException, IOException {
    GeneratorOptions opt = Main.parseArgs(
        new String[]{"--modify=false", "--output_dir=mango", "--a_few_files", "--parallel_deps"});
    assertEquals(opt.modificationMode, false);
    assertEquals(opt.outputDir, "mango");
    assertEquals(opt.aFewFiles, true);
    assertEquals(opt.manyFiles, false);
    assertEquals(opt.longChainedDeps, false);
    assertEquals(opt.parallelDeps, true);
  }
}
