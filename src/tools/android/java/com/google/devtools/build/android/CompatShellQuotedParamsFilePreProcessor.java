// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.io.PushbackReader;
import java.io.Reader;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Emulates the behavior of the ShellQuotedParamsFilePreProcessor class from Bazel.
 *
 * <p>This class purely emulates the quote escaping/unescaping that
 * ShellQuotedParamsFilePreProcessor does. It is intended to be used in ResourceProcessorBusyBox
 * (and affiliated tools) in conjunction with JCommander's Parameter annotations instead of the
 * Bazel-specific OptionsParser. There is no guarantee that this class will behave 100% identically
 * to ShellQuotedParamsFilePreProcessor.
 */
public class CompatShellQuotedParamsFilePreProcessor {
  private FileSystem fs;
  static final String UNFINISHED_QUOTE_MESSAGE_FORMAT = "Unfinished quote %s at %s";

  public CompatShellQuotedParamsFilePreProcessor(FileSystem fs) {
    this.fs = fs;
  }

  public List<String> preProcess(List<String> args) throws CompatOptionsParsingException {
    if (!args.isEmpty() && args.get(0).startsWith("@")) {
      if (args.size() > 1) {
        throw new CompatOptionsParsingException(
            String.format("A params file must be the only argument: %s", args), args.get(0));
      }
      Path path = fs.getPath(args.get(0).substring(1));
      try {
        return parse(path);
      } catch (RuntimeException | IOException e) {
        throw new CompatOptionsParsingException(
            String.format("Error reading params file: %s %s", path, e.getMessage()),
            args.get(0),
            e);
      }
    }
    return args;
  }

  public List<String> parse(Path paramsFile) throws IOException {
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

    @Nullable
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
        throw new IOException(String.format(UNFINISHED_QUOTE_MESSAGE_FORMAT, "'", quoteStart));
      }
      return arg.toString();
    }
  }
}
