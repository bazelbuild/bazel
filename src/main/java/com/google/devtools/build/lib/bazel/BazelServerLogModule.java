// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel;

import com.google.devtools.build.lib.runtime.BlazeModule;
import java.util.logging.Handler;
import java.util.logging.Logger;

/** Provides configuration flags and hooks for the Bazel server log handler. */
public class BazelServerLogModule extends BlazeModule {
  @Override
  public void commandComplete() {
    // Flush the server log after each command.
    for (Logger logger = Logger.getLogger(BazelServerLogModule.class.getName());
        logger != null;
        logger = logger.getParent()) {
      for (Handler handler : logger.getHandlers()) {
        if (handler != null) {
          handler.flush();
        }
      }
    }
  }
}
