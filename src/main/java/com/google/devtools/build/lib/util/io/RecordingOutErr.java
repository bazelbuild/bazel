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
package com.google.devtools.build.lib.util.io;

import java.io.ByteArrayOutputStream;
import java.io.UnsupportedEncodingException;

/**
 * An implementation of {@link OutErr} that captures all out / err output and
 * makes it available as ISO-8859-1 strings. Useful for implementing test
 * cases that assert particular output.
 */
public class RecordingOutErr extends OutErr {

  public RecordingOutErr() {
    super(new ByteArrayOutputStream(), new ByteArrayOutputStream());
  }

  public RecordingOutErr(ByteArrayOutputStream out, ByteArrayOutputStream err) {
    super(out, err);
  }

  /**
   * Reset the captured content; that is, reset the out / err buffers.
   */
  public void reset() {
    getOutputStream().reset();
    getErrorStream().reset();
  }

  /**
   * Interprets the captured out content as an {@code ISO-8859-1} encoded
   * string.
   */
  public String outAsLatin1() {
    try {
      return getOutputStream().toString("ISO-8859-1");
    } catch (UnsupportedEncodingException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Interprets the captured err content as an {@code ISO-8859-1} encoded
   * string.
   */
  public String errAsLatin1() {
    try {
      return getErrorStream().toString("ISO-8859-1");
    } catch (UnsupportedEncodingException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Returns true if any output was recorded.
   */
  public boolean hasRecordedOutput() {
    return getOutputStream().size() > 0 || getErrorStream().size() > 0;
  }

  @Override
  public String toString() {
    String out = outAsLatin1();
    String err = errAsLatin1();
    return "" + ((out.length() > 0) ? ("stdout: " + out + "\n") : "")
              + ((err.length() > 0) ? ("stderr: " + err) : "");
  }

  @Override
  public ByteArrayOutputStream getOutputStream() {
    return (ByteArrayOutputStream) super.getOutputStream();
  }

  @Override
  public ByteArrayOutputStream getErrorStream() {
    return (ByteArrayOutputStream) super.getErrorStream();
  }

}
