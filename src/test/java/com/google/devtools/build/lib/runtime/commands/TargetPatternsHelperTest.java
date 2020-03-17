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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.TargetPatternsHelper.TargetPatternsHelperException;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests {@link TargetPatternsHelper}. */
@RunWith(JUnit4.class)
public class TargetPatternsHelperTest {

  private CommandEnvironment env;
  private Scratch scratch;
  private OptionsParser options;

  @Before
  public void setUp() throws Exception {
    options = OptionsParser.builder().optionsClasses(BuildRequestOptions.class).build();
    scratch = new Scratch();
    BlazeRuntime runtime =
        new BlazeRuntime.Builder()
            .setFileSystem(scratch.getFileSystem())
            .setProductName(TestConstants.PRODUCT_NAME)
            .setServerDirectories(
                new ServerDirectories(
                    scratch.resolve("/install"),
                    scratch.resolve("/base"),
                    scratch.resolve("/userRoot")))
            .setStartupOptionsProvider(
                OptionsParser.builder().optionsClasses(BlazeServerStartupOptions.class).build())
            .build();
    env = Mockito.mock(CommandEnvironment.class);
    when(env.getWorkingDirectory()).thenReturn(scratch.resolve("wd"));
    when(env.getRuntime()).thenReturn(runtime);
  }

  @Test
  public void testEmpty() throws TargetPatternsHelperException {
    // tests when no residue and no --target_pattern_file are set
    assertThat(TargetPatternsHelper.readFrom(env, options)).isEmpty();
  }

  @Test
  public void testTargetPatternFile() throws Exception {
    scratch.file("/wd/patterns.txt", "//some/...\n//patterns");
    options.parse("--target_pattern_file=patterns.txt");

    assertThat(TargetPatternsHelper.readFrom(env, options))
        .isEqualTo(ImmutableList.of("//some/...", "//patterns"));
  }

  @Test
  public void testNoTargetPatternFile() throws TargetPatternsHelperException {
    ImmutableList<String> patterns = ImmutableList.of("//some/...", "//patterns");
    options.setResidue(patterns, ImmutableList.of());

    assertThat(TargetPatternsHelper.readFrom(env, options)).isEqualTo(patterns);
  }

  @Test
  public void testSpecifyPatternAndFileThrows() throws OptionsParsingException {
    options.parse("--target_pattern_file=patterns.txt");
    options.setResidue(ImmutableList.of("//some:pattern"), ImmutableList.of());

    TargetPatternsHelperException expected =
        assertThrows(
            TargetPatternsHelperException.class, () -> TargetPatternsHelper.readFrom(env, options));

    assertThat(expected)
        .hasMessageThat()
        .isEqualTo(
            "Command-line target pattern and --target_pattern_file cannot both be specified");
  }

  @Test
  public void testSpecifyNonExistingFileThrows() throws OptionsParsingException {
    options.parse("--target_pattern_file=patterns.txt");

    TargetPatternsHelperException expected =
        assertThrows(
            TargetPatternsHelperException.class, () -> TargetPatternsHelper.readFrom(env, options));

    assertThat(expected)
        .hasMessageThat()
        .matches("I/O error reading from .*patterns.txt.*\\(No such file or directory\\)");
  }
}
