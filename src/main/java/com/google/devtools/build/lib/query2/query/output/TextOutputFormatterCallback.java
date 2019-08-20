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

package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.Writer;

/** Abstract class supplying a {@link PrintStream} to implementations, flushing it on close. */
abstract class TextOutputFormatterCallback<T> extends OutputFormatterCallback<T> {
  protected Writer writer;

  @SuppressWarnings("DefaultCharset")
  TextOutputFormatterCallback(OutputStream out) {
    // This code intentionally uses the platform default encoding.
    this.writer = new BufferedWriter(new OutputStreamWriter(out));
  }

  @Override
  public void close(boolean failFast) throws IOException {
    writer.flush();
  }
}