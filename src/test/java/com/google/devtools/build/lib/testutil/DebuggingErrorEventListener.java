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

package com.google.devtools.build.lib.testutil;

import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;

import java.io.PrintStream;

/**
 * Prints all errors and warnings to {@link System#out}.
 */
public class DebuggingErrorEventListener implements ErrorEventListener {

  private PrintStream out;

  public DebuggingErrorEventListener() {
    this.out = System.out;
  }

  private void print(EventKind kind, Location location, String message) {
    if (location != null) {
      out.println(kind + " " + location + ": " + message);
    } else {
      out.println(kind + " " + message);
    }
  }

  @Override
  public void warn(Location location, String message) {
    print(EventKind.WARNING, location, message);
  }

  @Override
  public void error(Location location, String message) {
    print(EventKind.ERROR, location, message);
  }

  @Override
  public void info(Location location, String message) {
    print(EventKind.INFO, location, message);
  }

  @Override
  public void progress(Location location, String message) {
    print(EventKind.PROGRESS, location, message);
  }

  @Override
  public void report(EventKind kind, Location location, String message) {
    print(kind, location, message);
  }

  @Override
  public boolean showOutput(String tag) {
    return true;
  }
}
