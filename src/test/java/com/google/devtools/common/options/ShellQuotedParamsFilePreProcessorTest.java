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
package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.jimfs.Jimfs;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ShellQuotedParamsFilePreProcessor}.
 */
@RunWith(JUnit4.class)
public class ShellQuotedParamsFilePreProcessorTest {

  private FileSystem fileSystem;
  private Path paramsFile;
  private ParamsFilePreProcessor paramsFilePreProcessor;

  @Before
  public void setup() {
    fileSystem = Jimfs.newFileSystem();
    paramsFile = fileSystem.getPath("paramsFile");
    paramsFilePreProcessor = new ShellQuotedParamsFilePreProcessor(fileSystem);
  }

  @Test
  public void testUnquotedWhitespaceSeparated() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("arg1 arg2  arg3\targ4\\ arg5"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("arg1", "arg2", "", "arg3", "arg4\\", "arg5").inOrder();
  }

  @Test
  public void testUnquotedNewlineSeparated() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("arg1\narg2\rarg3\r\narg4\n\rarg5"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("arg1", "arg2", "arg3", "arg4", "", "arg5").inOrder();
  }

  @Test
  public void testQuotedWhitespaceSeparated() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("'arg1' 'arg2' '' 'arg3'\t'arg4'\\ 'arg5'"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("arg1", "arg2", "", "arg3", "arg4\\", "arg5").inOrder();
  }

  @Test
  public void testQuotedNewlineSeparated() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("'arg1'\n'arg2'\r'arg3'\r\n'arg4'\n''\r'arg5'"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("arg1", "arg2", "arg3", "arg4", "", "arg5").inOrder();
  }

  @Test
  public void testQuotedContainingWhitespace() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("'arg1 arg2  arg3\targ4'"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("arg1 arg2  arg3\targ4").inOrder();
  }

  @Test
  public void testQuotedContainingNewline() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("'arg1\narg2\rarg3\r\narg4\n\rarg5'"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("arg1\narg2\rarg3\r\narg4\n\rarg5").inOrder();
  }

  @Test
  public void testQuotedContainingQuotes() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("'ar'\\''g1'", "'a'\\''rg'\\''2'"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("ar'g1", "a'rg'2").inOrder();
  }

  @Test
  public void testPartialQuoting() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("ar'g1 ar'g2 arg3"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("arg1 arg2", "arg3").inOrder();
  }

  @Test
  public void testUnfinishedQuote() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("ar'g1"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    OptionsParsingException expected =
        assertThrows(
            OptionsParsingException.class,
            () -> preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile)));
    assertThat(expected)
        .hasMessageThat()
        .isEqualTo(
            String.format(
                ParamsFilePreProcessor.ERROR_MESSAGE_FORMAT,
                paramsFile,
                String.format(ParamsFilePreProcessor.UNFINISHED_QUOTE_MESSAGE_FORMAT, "'", 2)));
  }

  @Test
  public void testUnquotedEscapedQuote() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("ar\\'g1'"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args).containsExactly("ar\\g1").inOrder();
  }

  private static List<String> preProcess(ParamsFilePreProcessor preProcessor, List<String> args)
      throws OptionsParsingException {
    return Lists.transform(
        preProcessor.preProcess(OptionsParser.ArgAndFallbackData.wrapWithFallbackData(args, null)),
        argAndFallbackData -> argAndFallbackData.arg);
  }
}
