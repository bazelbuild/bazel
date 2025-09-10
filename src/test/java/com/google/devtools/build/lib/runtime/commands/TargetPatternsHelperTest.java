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
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.TargetPatternsHelper.TargetPatternsHelperException;
import com.google.devtools.build.lib.runtime.events.InputFileEvent;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns.Code;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Set;
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
  private MockEventBus mockEventBus;

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
    mockEventBus = new MockEventBus();
    env = Mockito.mock(CommandEnvironment.class);
    when(env.getWorkingDirectory()).thenReturn(scratch.resolve("wd"));
    when(env.getRuntime()).thenReturn(runtime);
    when(env.getEventBus()).thenReturn(mockEventBus);
  }

  @Test
  public void testEmpty() throws TargetPatternsHelperException {
    // tests when no residue and no --target_pattern_file are set
    assertThat(TargetPatternsHelper.readFrom(env, options)).isEmpty();
  }

  @Test
  public void testTargetPatternFile() throws Exception {
    Path targetPatternFilePath = scratch.file("/wd/patterns.txt", "//some/...\n//patterns");
    options.parse("--target_pattern_file=patterns.txt");

    assertThat(TargetPatternsHelper.readFrom(env, options))
        .isEqualTo(ImmutableList.of("//some/...", "//patterns"));
    assertThat(mockEventBus.inputFileEvents)
        .containsExactly(
            InputFileEvent.create(
                /* type= */ "target_pattern_file", targetPatternFilePath.getFileSize()));
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

    String message =
        "Only one of command-line target patterns, --target_pattern_file, --query, "
            + "or --query_file may be specified";
    assertThat(expected).hasMessageThat().isEqualTo(message);
    assertThat(expected.getFailureDetail())
        .isEqualTo(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setTargetPatterns(
                    TargetPatterns.newBuilder()
                        .setCode(Code.TARGET_PATTERN_FILE_WITH_COMMAND_LINE_PATTERN))
                .build());
  }

  @Test
  public void testSpecifyNonExistingFileThrows() throws OptionsParsingException {
    options.parse("--target_pattern_file=patterns.txt");

    TargetPatternsHelperException expected =
        assertThrows(
            TargetPatternsHelperException.class, () -> TargetPatternsHelper.readFrom(env, options));

    String regex = "I/O error reading from .*patterns.txt.*\\(No such file or directory\\)";
    assertThat(expected).hasMessageThat().matches(regex);
    assertThat(expected.getFailureDetail().getMessage()).matches(regex);
    assertThat(expected.getFailureDetail().hasTargetPatterns()).isTrue();
    assertThat(expected.getFailureDetail().getTargetPatterns().getCode())
        .isEqualTo(Code.TARGET_PATTERN_FILE_READ_FAILURE);
  }

  @Test
  public void testSpecifyMultipleOptionsThrows() throws OptionsParsingException {
    options.parse("--target_pattern_file=patterns.txt", "--query=deps(//...)");

    TargetPatternsHelperException expected =
        assertThrows(
            TargetPatternsHelperException.class, () -> TargetPatternsHelper.readFrom(env, options));

    String message =
        "Only one of command-line target patterns, --target_pattern_file, --query, "
            + "or --query_file may be specified";
    assertThat(expected).hasMessageThat().isEqualTo(message);
    assertThat(expected.getFailureDetail())
        .isEqualTo(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setTargetPatterns(
                    TargetPatterns.newBuilder()
                        .setCode(Code.TARGET_PATTERN_FILE_WITH_COMMAND_LINE_PATTERN))
                .build());
  }

  @Test
  public void testSpecifyQueryAndPatternThrows() throws OptionsParsingException {
    options.parse("--query=deps(//...)");
    options.setResidue(ImmutableList.of("//some:pattern"), ImmutableList.of());

    TargetPatternsHelperException expected =
        assertThrows(
            TargetPatternsHelperException.class, () -> TargetPatternsHelper.readFrom(env, options));

    String message =
        "Only one of command-line target patterns, --target_pattern_file, --query, "
            + "or --query_file may be specified";
    assertThat(expected).hasMessageThat().isEqualTo(message);
  }

  @Test
  public void testQueryFileWithNonExistingFileThrows() throws OptionsParsingException {
    options.parse("--query_file=query.txt");

    TargetPatternsHelperException expected =
        assertThrows(
            TargetPatternsHelperException.class, () -> TargetPatternsHelper.readFrom(env, options));

    String regex = "I/O error reading from .*query.txt.*\\(No such file or directory\\)";
    assertThat(expected).hasMessageThat().matches(regex);
    assertThat(expected.getFailureDetail().hasTargetPatterns()).isTrue();
    assertThat(expected.getFailureDetail().getTargetPatterns().getCode())
        .isEqualTo(Code.TARGET_PATTERN_FILE_READ_FAILURE);
  }

  private static class MockEventBus extends EventBus {
    final Set<InputFileEvent> inputFileEvents = Sets.newConcurrentHashSet();

    @Override
    public void post(Object event) {
      inputFileEvents.add((InputFileEvent) event);
    }
  }
}
