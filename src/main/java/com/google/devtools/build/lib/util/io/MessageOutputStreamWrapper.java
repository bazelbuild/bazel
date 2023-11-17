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

import com.google.common.base.Preconditions;
import com.google.protobuf.Message;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/** Creates a MessageOutputStream from an OutputStream. */
public class MessageOutputStreamWrapper {
  /** Outputs the messages in delimited protobuf binary format. */
  public static class BinaryOutputStreamWrapper<T extends Message>
      implements MessageOutputStream<T> {
    private final OutputStream stream;

    public BinaryOutputStreamWrapper(OutputStream stream) {
      this.stream = Preconditions.checkNotNull(stream);
    }

    @Override
    public void write(T m) throws IOException {
      Preconditions.checkNotNull(m);
      m.writeDelimitedTo(stream);
    }

    @Override
    public void close() throws IOException {
      stream.close();
    }
  }

  /** Outputs the messages in JSON text format. */
  public static class JsonOutputStreamWrapper<T extends Message> implements MessageOutputStream<T> {
    private final OutputStream stream;
    private final JsonFormat.Printer printer = JsonFormat.printer().includingDefaultValueFields();

    public JsonOutputStreamWrapper(OutputStream stream) {
      Preconditions.checkNotNull(stream);
      this.stream = stream;
    }

    @Override
    public void write(T m) throws IOException {
      Preconditions.checkNotNull(m);
      stream.write(printer.print(m).getBytes(StandardCharsets.UTF_8));
    }

    @Override
    public void close() throws IOException {
      stream.close();
    }
  }
}
