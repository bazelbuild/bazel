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
 * Tests {@link UnquotedParamsFilePreProcessor}.
 */
@RunWith(JUnit4.class)
public class UnquotedParamsFilePreProcessorTest {
  private Path paramsFile;
  private ParamsFilePreProcessor paramsFilePreProcessor;

  @Before
  public void setup() {
    FileSystem fileSystem = Jimfs.newFileSystem();
    paramsFile = fileSystem.getPath("paramsFile");
    paramsFilePreProcessor = new UnquotedParamsFilePreProcessor(fileSystem);
  }

  @Test
  public void testNewlines() throws IOException, OptionsParsingException {
    Files.write(
        paramsFile,
        ImmutableList.of("arg1\narg2\rarg3\r\narg4 arg5\targ6\n\rarg7\\ arg8"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);
    List<String> args = preProcess(paramsFilePreProcessor, ImmutableList.of("@" + paramsFile));
    assertThat(args)
        .containsExactly("arg1", "arg2", "arg3", "arg4 arg5\targ6", "", "arg7\\ arg8")
        .inOrder();
  }

  private static List<String> preProcess(ParamsFilePreProcessor preProcessor, List<String> args)
      throws OptionsParsingException {
    return Lists.transform(
        preProcessor.preProcess(
            Lists.transform(
                args,
                arg -> new OptionsParserImpl.ArgAndFallbackData(arg, /* fallbackData= */ null))),
        argAndFallbackData -> argAndFallbackData.arg);
  }
}
