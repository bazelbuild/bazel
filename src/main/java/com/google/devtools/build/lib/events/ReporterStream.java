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

package com.google.devtools.build.lib.events;

import com.google.devtools.build.lib.util.Preconditions;
import java.io.OutputStream;
import java.util.Arrays;

/**
 * An OutputStream that delegates all writes to a Reporter.
 */
public final class ReporterStream extends OutputStream {
  private final EventHandler reporter;
  private final EventKind eventKind;

  public ReporterStream(EventHandler reporter, EventKind eventKind) {
    this.reporter = Preconditions.checkNotNull(reporter);
    this.eventKind = Preconditions.checkNotNull(eventKind);
  }

  @Override
  public void close() {
    // NOP.
  }

  @Override
  public void flush() {
    // NOP.
  }

  @Override
  public void write(int b) {
    reporter.handle(Event.of(eventKind, null, new byte[] { (byte) b }));
  }

  @Override
  public void write(byte[] bytes) {
    write(bytes, 0, bytes.length);
  }

  @Override
  public void write(byte[] bytes, int offset, int len) {
    reporter.handle(Event.of(eventKind, null, Arrays.copyOfRange(bytes, offset, offset + len)));
  }
}
