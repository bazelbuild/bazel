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
package com.google.devtools.build.lib.analysis.actions;

import com.google.protobuf.AbstractMessageLite;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;

/** A {@link DeterministicWriter} wrapping an {@link AbstractMessageLite} supplier. */
public class ProtoDeterministicWriter implements DeterministicWriter {
  private final MessageSupplier messageSupplier;

  /** Constructs a {@link ProtoDeterministicWriter} with an eagerly constructed message. */
  public ProtoDeterministicWriter(AbstractMessageLite<?, ?> message) {
    this.messageSupplier = () -> message;
  }

  /**
   * Constructs a {@link ProtoDeterministicWriter} with the given supplier. The supplier may be
   * called multiple times.
   */
  public ProtoDeterministicWriter(MessageSupplier supplier) {
    this.messageSupplier = supplier;
  }

  @Override
  public void writeOutputFile(OutputStream out) throws IOException {
    messageSupplier.getMessage().writeTo(out);
  }

  @Override
  public ByteString getBytes() throws IOException {
    return messageSupplier.getMessage().toByteString();
  }

  /** Supplies an {@link AbstractMessageLite}, possibly throwing {@link IOException}. */
  @FunctionalInterface
  public interface MessageSupplier {
    AbstractMessageLite<?, ?> getMessage() throws IOException;
  }
}
