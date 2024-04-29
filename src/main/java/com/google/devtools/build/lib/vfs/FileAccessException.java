// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;

import com.google.devtools.build.lib.skyframe.serialization.LeafDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.LeafSerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * An IOException subclass that is thrown when a file system access is denied. The message is
 * generally "Permission denied".
 */
public class FileAccessException extends IOException {
  /**
   * Constructs an <code>FileAccessException</code> with the specified detail
   * message. The error message string <code>s</code> can later be
   * retrieved by the <code>{@link java.lang.Throwable#getMessage}</code>
   * method of class <code>java.lang.Throwable</code>.
   *
   * @param s the detail message.
   */
  public FileAccessException(String s) {
    super(s);
  }

  /**
   * Codec for {@link FileAccessException}.
   *
   * <p>{@link com.google.devtools.build.lib.skyframe.serialization.AutoRegistry} excludes the
   * entire com.google.devtools.build.lib.vfs java package from having DynamicCodec support.
   * Therefore, we need to provide our own codec for @link FileAccessException}.
   */
  @SuppressWarnings("unused") // found by CLASSPATH-scanning magic
  private static class Codec extends LeafObjectCodec<FileAccessException> {
    @Override
    public Class<? extends FileAccessException> getEncodedClass() {
      return FileAccessException.class;
    }

    @Override
    public void serialize(
        LeafSerializationContext context, FileAccessException fae, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      String message = fae.getMessage();
      if (message == null) {
        codedOut.writeBoolNoTag(false);
        return;
      }
      codedOut.writeBoolNoTag(true);
      context.serializeLeaf(message, stringCodec(), codedOut);
    }

    @Override
    public FileAccessException deserialize(
        LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      boolean hasMessage = codedIn.readBool();
      if (!hasMessage) {
        return new FileAccessException(null);
      }
      String message = context.deserializeLeaf(codedIn, stringCodec());
      return new FileAccessException(message);
    }
  }
}
