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
package com.google.devtools.build.lib.server.signal;

import sun.misc.Signal;

/** Class that can be extended to handle SIGINT in a custom way. */
public abstract class InterruptSignalHandler extends AbstractSignalHandler {
  private static final Signal SIGINT = new Signal("INT");

  /**
   * Constructs an InterruptSignalHandler instance.  Until the uninstall()
   * method is invoked, the delivery of a SIGINT signal to this process will
   * cause the run() method to be invoked in another thread.
   */
  protected InterruptSignalHandler() {
    super(SIGINT);
  }
}
