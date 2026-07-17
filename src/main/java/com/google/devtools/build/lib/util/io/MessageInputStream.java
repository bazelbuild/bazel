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

import com.google.protobuf.Message;
import java.io.IOException;
import javax.annotation.Nullable;

/** A variation of InputStream for protobuf messages. */
public interface MessageInputStream<T extends Message> extends AutoCloseable {
  /** Reads a protobuf message from the underlying stream, or null if there are no more messages. */
  @Nullable
  T read() throws IOException;

  /** Closes the underlying stream. Any following reads will fail. */
  @Override
  void close() throws IOException;
}
