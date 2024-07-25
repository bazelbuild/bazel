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
import com.google.devtools.common.options.OptionsParser.ArgAndFallbackData;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ParamsFilePreProcessor}.
 */
@RunWith(JUnit4.class)
public class ParamsFilePreProcessorTest {
  private static final ImmutableList<String> PARAM_FILE_ARGS =
      ImmutableList.of("params", "file", "args");

  private static class MockParamsFilePreProcessor extends ParamsFilePreProcessor {

    MockParamsFilePreProcessor(FileSystem fs) {
      super(fs);
    }

    @Override
    protected List<String> parse(Path paramsFile) throws IOException, OptionsParsingException {
      return PARAM_FILE_ARGS;
    }
  }

  private FileSystem fileSystem;
  private ParamsFilePreProcessor paramsFilePreProcessor;

  @Before
  public void setup() {
    fileSystem = Jimfs.newFileSystem();
    paramsFilePreProcessor = new MockParamsFilePreProcessor(fileSystem);
  }

  @Test
  public void testNoArgs() throws OptionsParsingException {
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of());
    assertThat(args).isEmpty();
  }

  @Test
  public void testNoParamsFile() throws OptionsParsingException {
    List<String> rawArgs = ImmutableList.of("--foo", "foo val");
    List<String> args = preProcess(paramsFilePreProcessor, rawArgs);
    assertThat(args).containsExactlyElementsIn(rawArgs).inOrder();
  }

  @Test
  public void testParamsFileNotFirst() throws OptionsParsingException {
    List<String> rawArgs = ImmutableList.of("--foo", "foo val", "@paramsFile");
    List<String> args = preProcess(paramsFilePreProcessor, rawArgs);
    assertThat(args).containsExactlyElementsIn(rawArgs).inOrder();
  }

  @Test
  public void testTooManyArgs() throws OptionsParsingException {
    List<String> rawArgs = ImmutableList.of("@paramsFile", "--foo", "foo val");
    OptionsParsingException expected =
        assertThrows(
            OptionsParsingException.class, () -> preProcess(paramsFilePreProcessor, rawArgs));
    assertThat(expected)
        .hasMessageThat()
        .isEqualTo(
            String.format(ParamsFilePreProcessor.TOO_MANY_ARGS_ERROR_MESSAGE_FORMAT, rawArgs));
  }

  @Test
  public void testExceptionDuringParsing() throws OptionsParsingException {
    ParamsFilePreProcessor exceptionParser = new ParamsFilePreProcessor(fileSystem) {
      @Override
      protected List<String> parse(Path paramsFile) throws IOException, OptionsParsingException {
        throw new IOException("Error parsing " + paramsFile);
      }
    };
    String paramsFileName = "paramsFile";
    Path paramsFile = fileSystem.getPath(paramsFileName);
    OptionsParsingException expected =
        assertThrows(
            OptionsParsingException.class,
            () -> preProcess(exceptionParser, ImmutableList.of("@" + paramsFileName)));
    assertThat(expected)
        .hasMessageThat()
        .isEqualTo(
            String.format(
                ParamsFilePreProcessor.ERROR_MESSAGE_FORMAT,
                paramsFile,
                "Error parsing " + paramsFileName));
  }

  @Test
  public void testParamsFile() throws OptionsParsingException {
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@paramsFile"));
    assertThat(args).containsExactlyElementsIn(PARAM_FILE_ARGS).inOrder();
  }

  private static List<String> preProcess(ParamsFilePreProcessor preProcessor, List<String> args)
      throws OptionsParsingException {
    return Lists.transform(
        preProcessor.preProcess(ArgAndFallbackData.wrapWithFallbackData(args, null)),
        argAndFallbackData -> argAndFallbackData.arg);
  }
}
