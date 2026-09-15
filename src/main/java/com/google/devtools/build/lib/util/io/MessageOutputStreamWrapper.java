// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.protobuf.Message;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.OutputStream;

/** Creates a MessageOutputStream from an OutputStream. */
public class MessageOutputStreamWrapper {

  private MessageOutputStreamWrapper() {}

  /** Writes the messages in length-delimited protobuf wire format. */
  public static class BinaryOutputStreamWrapper<T extends Message>
      implements MessageOutputStream<T> {
    private final OutputStream stream;

    public BinaryOutputStreamWrapper(OutputStream stream) {
      this.stream = checkNotNull(stream);
    }

    @Override
    public void write(T m) throws IOException {
      checkNotNull(m);
      m.writeDelimitedTo(stream);
    }

    @Override
    public void close() throws IOException {
      stream.close();
    }
  }

  /** Writes the messages in concatenated JSON text format. */
  public static class JsonOutputStreamWrapper<T extends Message> implements MessageOutputStream<T> {
    private static final JsonFormat.Printer PRINTER =
        JsonFormat.printer().alwaysPrintFieldsWithNoPresence();

    private final OutputStream stream;

    public JsonOutputStreamWrapper(OutputStream stream) {
      this.stream = checkNotNull(stream);
    }

    @Override
    public void write(T m) throws IOException {
      checkNotNull(m);
      stream.write(PRINTER.print(m).getBytes(UTF_8));
    }

    @Override
    public void close() throws IOException {
      stream.close();
    }
  }
}
