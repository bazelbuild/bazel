// Copyright 2014 Google Inc. All rights reserved.
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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import java.io.OutputStream;

/**
 * An OutputStream that delegates all writes to a Reporter.
 */
public final class ReporterStream extends OutputStream {

  private final EventHandler reporter;
  private final EventKind eventKind;

  public ReporterStream(EventHandler reporter, EventKind eventKind) {
    this.reporter = reporter;
    this.eventKind = eventKind;
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
    reporter.handle(new Event(eventKind, null, new byte[] { (byte) b }));
  }

  @Override
  public void write(byte[] bytes) {
    reporter.handle(new Event(eventKind, null, bytes));
  }

  @Override
  public void write(byte[] bytes, int offset, int len) {
    reporter.handle(new Event(eventKind, null, new String(bytes, offset, len, ISO_8859_1)));
  }
}
