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

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.io.PushbackReader;
import java.io.Reader;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link ParamsFilePreProcessor} that processes a parameter file using the {@code
 * com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType.SHELL_QUOTED} format. This
 * format assumes each parameter is separated by whitespace and is quoted using singe quotes
 * ({@code '}) if it contains any special characters or is an empty string.
 */
public class ShellQuotedParamsFilePreProcessor extends ParamsFilePreProcessor {

  public ShellQuotedParamsFilePreProcessor(FileSystem fs) {
    super(fs);
  }

  @Override
  protected List<String> parse(Path paramsFile) throws IOException {
    List<String> args = new ArrayList<>();
    try (ShellQuotedReader reader =
        new ShellQuotedReader(Files.newBufferedReader(paramsFile, UTF_8))) {
      String arg;
      while ((arg = reader.readArg()) != null) {
        args.add(arg);
      }
    }
    return args;
  }

  private static class ShellQuotedReader implements AutoCloseable {

    private final PushbackReader reader;
    private int position = -1;

    public ShellQuotedReader(Reader reader) {
      this.reader = new PushbackReader(reader, 10);
    }

    private char read() throws IOException {
      int value = reader.read();
      position++;
      return (char) value;
    }

    private void unread(char value) throws IOException {
      reader.unread(value);
      position--;
    }

    private boolean hasNext() throws IOException {
      char value = read();
      boolean hasNext = value != (char) -1;
      unread(value);
      return hasNext;
    }

    @Override
    public void close() throws IOException {
      reader.close();
    }

    public String readArg() throws IOException {
      if (!hasNext()) {
        return null;
      }

      StringBuilder arg = new StringBuilder();

      int quoteStart = -1;
      boolean quoted = false;
      char current;

      while ((current = read()) != (char) -1) {
        if (quoted) {
          if (current == '\'') {
            StringBuilder escapedQuoteRemainder =
                new StringBuilder().append(read()).append(read()).append(read());
            if (escapedQuoteRemainder.toString().equals("\\''")) {
              arg.append("'");
            } else {
              for (char c : escapedQuoteRemainder.reverse().toString().toCharArray()) {
                unread(c);
              }
              quoted = false;
              quoteStart = -1;
            }
          } else {
            arg.append(current);
          }
        } else {
          if (current == '\'') {
            quoted = true;
            quoteStart = position;
          } else if (current == '\r') {
            char next = read();
            if (next == '\n') {
              return arg.toString();
            } else {
              unread(next);
              return arg.toString();
            }
          } else if (Character.isWhitespace(current)) {
            return arg.toString();
          } else {
            arg.append(current);
          }
        }
      }
      if (quoted) {
        throw new IOException(
            String.format(UNFINISHED_QUOTE_MESSAGE_FORMAT, "'", quoteStart));
      }
      return arg.toString();
    }
  }
}
