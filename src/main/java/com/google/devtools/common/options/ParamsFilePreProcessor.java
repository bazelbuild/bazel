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

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Defines an {@link ArgsPreProcessor} that will determine if the arguments list contains a "params"
 * file that contains a list of options to be parsed.
 *
 * <p>Params files are used when the argument list of {@link Option} exceed the shells commandline
 * length. A params file argument is defined as a path starting with @. It will also be the only
 * entry in an argument list.
 */
public class ParamsFilePreProcessor implements ArgsPreProcessor {

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
   * @return A list of areguments suitable for parsing.
   * @throws OptionsParsingException if the path does not exist.
   */
  @Override
  public List<String> preProcess(List<String> args) throws OptionsParsingException {
    if (!args.isEmpty() && args.get(0).startsWith("@")) {
      if (args.size() > 1) {
        throw new OptionsParsingException(
            String.format(TOO_MANY_ARGS_ERROR_MESSAGE_FORMAT, args), args.get(0));
      }
      Path path = fs.getPath(args.get(0).substring(1));
      try (Reader params = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
        List<String> newArgs = new ArrayList<>();
        StringBuilder arg = new StringBuilder();
        CharIterator iterator = CharIterator.wrap(params);
        while (iterator.hasNext()) {
          char next = iterator.next();
          if (Character.isWhitespace(next) && !iterator.isInQuote() && !iterator.isEscaped()) {
            newArgs.add(arg.toString());
            arg = new StringBuilder();
          } else {
            arg.append(next);
          }
        }
        // If there is an arg in the buffer, add it.
        if (arg.length() > 0) {
          newArgs.add(arg.toString());
        }
        // If we're still in a quote by the end of the file, throw an error.
        if (iterator.isInQuote()) {
          throw new OptionsParsingException(
              String.format(ERROR_MESSAGE_FORMAT, path, iterator.getUnmatchedQuoteMessage()));
        }
        return newArgs;
      } catch (RuntimeException | IOException e) {
        throw new OptionsParsingException(
            String.format(ERROR_MESSAGE_FORMAT, path, e.getMessage()), args.get(0), e);
      }
    }
    return args;
  }

  // Doesn't implement iterator to avoid autoboxing and to throw exceptions.
  static class CharIterator {

    private final Reader reader;
    private int readerPosition = 0;
    private int singleQuoteStart = -1;
    private int doubleQuoteStart = -1;
    private boolean escaped = false;
    private char lastChar = (char) -1;

    public static CharIterator wrap(Reader reader) {
      return new CharIterator(reader);
    }

    public CharIterator(Reader reader) {
      this.reader = reader;
    }

    public boolean hasNext() throws IOException {
      return peek() != -1;
    }

    private int peek() throws IOException {
      reader.mark(1);
      int next = reader.read();
      reader.reset();
      return next;
    }

    public boolean isInQuote() {
      return singleQuoteStart != -1 || doubleQuoteStart != -1;
    }

    public boolean isEscaped() {
      return escaped;
    }

    public String getUnmatchedQuoteMessage() {
      StringBuilder message = new StringBuilder();
      if (singleQuoteStart != -1) {
        message.append(String.format(UNFINISHED_QUOTE_MESSAGE_FORMAT, "'", singleQuoteStart));
      }
      if (doubleQuoteStart != -1) {
        message.append(String.format(UNFINISHED_QUOTE_MESSAGE_FORMAT, "\"", doubleQuoteStart));
      }
      return message.toString();
    }

    public char next() throws IOException {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      char current = (char) reader.read();
      
      // check for \r\n line endings. If found, drop the \r for normalized parsing.
      if (current == '\r' && peek() == '\n') {
        current = (char) reader.read();
      }

      // check to see if the current position is escaped
      if (lastChar == '\\') {
        escaped = true;
      } else {
        escaped = false;
      }

      if (!escaped && current == '\'') {
        singleQuoteStart = singleQuoteStart == -1 ? readerPosition : -1;
      }
      if (!escaped && current == '"') {
        doubleQuoteStart = doubleQuoteStart == -1 ? readerPosition : -1;
      }

      readerPosition++;
      lastChar = current;
      return current;
    }
  }
}
