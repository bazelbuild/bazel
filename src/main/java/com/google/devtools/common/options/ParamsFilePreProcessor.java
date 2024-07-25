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

import com.google.common.collect.Lists;
import com.google.devtools.common.options.OptionsParserImpl.ArgAndFallbackData;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.List;

/**
 * Defines an {@link ArgsPreProcessor} that will determine if the arguments list contains a "params"
 * file that contains a list of options to be parsed.
 *
 * <p>Params files are used when the argument list of {@link Option} exceed the shells commandline
 * length. A params file argument is defined as a path starting with @. It will also be the only
 * entry in an argument list.
 */
public abstract class ParamsFilePreProcessor implements ArgsPreProcessor {

  static final String ERROR_MESSAGE_FORMAT = "Error reading params file: %s %s";

  static final String TOO_MANY_ARGS_ERROR_MESSAGE_FORMAT =
      "A params file must be the only argument: %s";

  static final String UNFINISHED_QUOTE_MESSAGE_FORMAT = "Unfinished quote %s at %s";

  private final FileSystem fs;

  ParamsFilePreProcessor(FileSystem fs) {
    this.fs = fs;
  }

  /**
   * Parses the param file path and replaces the arguments list with the contents if one exists.
   *
   * @param args A list of arguments that may contain @&lt;path&gt; to a params file.
   * @return A list of arguments suitable for parsing.
   * @throws OptionsParsingException if the path does not exist.
   */
  @Override
  public List<ArgAndFallbackData> preProcess(List<ArgAndFallbackData> args)
      throws OptionsParsingException {
    if (!args.isEmpty() && args.getFirst().arg.startsWith("@")) {
      if (args.size() > 1) {
        throw new OptionsParsingException(
            String.format(
                TOO_MANY_ARGS_ERROR_MESSAGE_FORMAT,
                Lists.transform(args, argAndFallbackData -> argAndFallbackData.arg)),
            args.getFirst().arg);
      }
      Path path = fs.getPath(args.getFirst().arg.substring(1));
      try {
        return Lists.transform(
            parse(path), arg -> new ArgAndFallbackData(arg, args.getFirst().fallbackData));
      } catch (RuntimeException | IOException e) {
        throw new OptionsParsingException(
            String.format(ERROR_MESSAGE_FORMAT, path, e.getMessage()), args.getFirst().arg, e);
      }
    }
    return args;
  }

  /**
   * Parses the paramsFile and returns a list of argument tokens to be further processed by the
   * {@link OptionsParser}.
   *
   * @param paramsFile The path of the params file to parse.
   * @return a list of argument tokens.
   * @throws IOException if there is an error reading paramsFile.
   * @throws OptionsParsingException if there is an error reading paramsFile.
   */
  protected abstract List<String> parse(Path paramsFile)
      throws IOException, OptionsParsingException;
}
