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
import java.util.ArrayList;

/** Creating a MessageOutputStream from an OutputStream */
public class MessageOutputStreamWrapper {
  /** Outputs the messages in binary format */
  public static class BinaryOutputStreamWrapper implements MessageOutputStream {
    private final OutputStream stream;

    public BinaryOutputStreamWrapper(OutputStream stream) {
      this.stream = Preconditions.checkNotNull(stream);
    }

    @Override
    public void write(Message m) throws IOException {
      Preconditions.checkNotNull(m);
      m.writeDelimitedTo(stream);
    }

    @Override
    public void close() throws IOException {
      stream.close();
    }
  }

  /** Outputs the messages in JSON text format */
  public static class JsonOutputStreamWrapper implements MessageOutputStream {
    private final OutputStream stream;
    private final JsonFormat.Printer printer = JsonFormat.printer().includingDefaultValueFields();

    public JsonOutputStreamWrapper(OutputStream stream) {
      Preconditions.checkNotNull(stream);
      this.stream = stream;
    }

    @Override
    public void write(Message m) throws IOException {
      Preconditions.checkNotNull(m);
      stream.write(printer.print(m).getBytes(StandardCharsets.UTF_8));
    }

    @Override
    public void close() throws IOException {
      stream.close();
    }
  }

  /** Outputs the messages in JSON text format */
  public static class MessageOutputStreamCollection implements MessageOutputStream {
    private final ArrayList<MessageOutputStream> streams = new ArrayList<>();

    public boolean isEmpty() {
      return streams.isEmpty();
    }

    public void addStream(MessageOutputStream m) {
      streams.add(m);
    }

    @Override
    public void write(Message m) throws IOException {
      for (MessageOutputStream stream : streams) {
        stream.write(m);
      }
    }

    @Override
    public void close() throws IOException {
      IOException firstException = null;
      for (MessageOutputStream stream : streams) {
        try {
          stream.close();
        } catch (IOException e) {
          if (firstException == null) {
            firstException = e;
          } else {
            firstException.addSuppressed(e);
          }
        }
      }
      if (firstException != null) {
        throw firstException;
      }
    }
  }
}
