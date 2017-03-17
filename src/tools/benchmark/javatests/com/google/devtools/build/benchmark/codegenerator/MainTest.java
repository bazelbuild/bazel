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

import static com.google.common.truth.Truth.assertThat;
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
      assertThat(e).hasMessage("--output_dir should not be empty.");
    }
  }

  @Test
  public void testParseArgsWrongMode() throws IOException {
    try {
      Main.parseArgs(new String[]{"--modify=mango"});
      fail("Should throw OptionsParsingException");
    } catch (OptionsParsingException e) {
      assertThat(e).hasMessage("While parsing option --modify=mango: 'mango' is not a boolean");
    }
  }

  @Test
  public void testParseArgsNoOutputDir() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[]{"--modify"});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessage("--output_dir should not be empty.");
    }
  }

  @Test
  public void testParseArgsOutputDirNonExists() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[]{"--modify", "--output_dir=mango"});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessage("--output_dir (mango) does not contain code for modification.");
    }
  }

  @Test
  public void testParseArgsNoType() throws OptionsParsingException, IOException {
    try {
      Main.parseArgs(new String[]{"--output_dir=mango"});
      fail("Should throw IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessage("No type of package is specified.");
    }
  }

  @Test
  public void testParseArgsCorrect() throws OptionsParsingException, IOException {
    GeneratorOptions opt = Main.parseArgs(
        new String[]{
            "--modify=false",
            "--output_dir=mango",
            "--project_name=AFewFiles",
            "--project_name=ParallelDeps"});
    assertThat(opt.modificationMode).isFalse();
    assertThat(opt.outputDir).isEqualTo("mango");
    assertThat(opt.projectNames).containsExactly("AFewFiles", "ParallelDeps");
  }
}
