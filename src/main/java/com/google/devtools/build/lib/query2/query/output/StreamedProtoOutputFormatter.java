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
package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.stream.StreamSupport;

/**
 * An output formatter that outputs a protocol buffer representation of a query result and outputs
 * the proto bytes to the output print stream. By taking the bytes and calling {@code mergeFrom()}
 * on a {@code Build.QueryResult} object the full result can be reconstructed.
 */
public class StreamedProtoOutputFormatter extends ProtoOutputFormatter {

  /**
   * The most bytes that protobuf delimited proto format will prepend to a proto message. See <a
   * href="https://github.com/protocolbuffers/protobuf/blob/c11033dc27c3e9c1913e45b62fb5d4c5b5644b3e/java/core/src/main/java/com/google/protobuf/AbstractMessageLite.java#L72">
   * <code>writeDelimitedTo</code></a> and <a
   * href="https://github.com/protocolbuffers/protobuf/blob/c11033dc27c3e9c1913e45b62fb5d4c5b5644b3e/java/core/src/main/java/com/google/protobuf/WireFormat.java#L28">
   * <code>MAX_VARINT32_SIZE</code></a>.
   *
   * <p>The value for int32 (used by {@code writeDelimitedTo} is actually 5, but we pick 10 just to
   * be safe.
   */
  private static final int MAX_BYTES_FOR_VARINT32_ENCODING = 10;

  @Override
  public String getName() {
    return "streamed_proto";
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      final OutputStream out, final QueryOptions options, LabelPrinter labelPrinter) {
    return new OutputFormatterCallback<Target>() {
      private final LabelPrinter ourLabelPrinter = labelPrinter;

      @Override
      public void processOutput(Iterable<Target> partialResult)
          throws IOException, InterruptedException {
        try {
          StreamSupport.stream(partialResult.spliterator(), /* parallel= */ true)
              .map(this::toProto)
              .map(StreamedProtoOutputFormatter::writeDelimited)
              // I imagine forEachOrdered hurts performance somewhat in some cases. While we may
              // not need to actually produce output in order, this code does not know whether
              // ordering was requested. So we just always write it in order, and hope performance
              // is OK.
              .forEachOrdered(this::writeToOutputStreamThreadSafe);
        } catch (WrappedIOException e) {
          throw e.getCause();
        } catch (WrappedInterruptedException e) {
          throw e.getCause();
        }
      }

      private Build.Target toProto(Target target) {
        try {
          return toTargetProtoBuffer(target, ourLabelPrinter);
        } catch (InterruptedException e) {
          throw new WrappedInterruptedException(e);
        }
      }

      private synchronized void writeToOutputStreamThreadSafe(ByteArrayOutputStream bout) {
        try {
          bout.writeTo(out);
        } catch (IOException e) {
          throw new WrappedIOException(e);
        }
      }
    };
  }

  private static ByteArrayOutputStream writeDelimited(Build.Target targetProtoBuffer) {
    try {
      var bout =
          new ByteArrayOutputStream(
              targetProtoBuffer.getSerializedSize() + MAX_BYTES_FOR_VARINT32_ENCODING);
      targetProtoBuffer.writeDelimitedTo(bout);
      return bout;
    } catch (IOException e) {
      throw new WrappedIOException(e);
    }
  }

  private static class WrappedIOException extends RuntimeException {
    private WrappedIOException(IOException cause) {
      super(cause);
    }

    @Override
    public IOException getCause() {
      return (IOException) super.getCause();
    }
  }

  private static class WrappedInterruptedException extends RuntimeException {
    private WrappedInterruptedException(InterruptedException cause) {
      super(cause);
    }

    @Override
    public InterruptedException getCause() {
      return (InterruptedException) super.getCause();
    }
  }
}
