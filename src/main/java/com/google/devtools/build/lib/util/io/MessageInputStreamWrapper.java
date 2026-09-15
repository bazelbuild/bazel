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
package com.google.devtools.build.lib.util.io;

import static com.google.common.base.Preconditions.checkNotNull;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.Message;
import com.google.protobuf.Parser;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.InputStream;
import java.util.Scanner;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** Creates a MessageInputStream from an OutputStream. */
public class MessageInputStreamWrapper {

  private MessageInputStreamWrapper() {}

  /** Reads the messages in length-delimited protobuf wire format. */
  public static class BinaryInputStreamWrapper<T extends Message> implements MessageInputStream<T> {
    private final InputStream stream;
    private final Parser<T> parser;

    @SuppressWarnings("unchecked")
    public BinaryInputStreamWrapper(InputStream stream, T defaultInstance) {
      this.stream = checkNotNull(stream);
      this.parser = (Parser<T>) defaultInstance.getParserForType();
    }

    @Override
    @Nullable
    public T read() throws IOException {
      return parser.parseDelimitedFrom(stream, ExtensionRegistry.getEmptyRegistry());
    }

    @Override
    public void close() throws IOException {
      stream.close();
    }
  }

  /** Reads the messages in concatenated JSON text format. */
  public static class JsonInputStreamWrapper<T extends Message> implements MessageInputStream<T> {
    private static final JsonFormat.Parser PARSER = JsonFormat.parser().ignoringUnknownFields();

    // The string `\n}{\n` is a reliable delimiter, but we must use lookbehind/lookahead to avoid
    // consuming the braces when tokenizing.
    private static final Pattern DELIMITER = Pattern.compile("(?<=\\n\\})(?=\\{\\n)");

    private final Scanner scanner;
    private final Supplier<Message.Builder> builderSupplier;

    public JsonInputStreamWrapper(InputStream stream, T defaultInstance) {
      this.scanner = new Scanner(checkNotNull(stream), UTF_8).useDelimiter(DELIMITER);
      this.builderSupplier = defaultInstance::newBuilderForType;
    }

    @Override
    @Nullable
    @SuppressWarnings("unchecked")
    public T read() throws IOException {
      if (!scanner.hasNext()) {
        return null;
      }
      Message.Builder builder = builderSupplier.get();
      PARSER.merge(scanner.next(), builder);
      return (T) builder.build();
    }

    @Override
    public void close() throws IOException {
      scanner.close();
    }
  }
}
