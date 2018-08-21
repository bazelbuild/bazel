// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.junit;

import org.junit.Rule;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Base test class for testing {@link TimeoutTestWatcher}; it is a base class since we want to
 * imitate the usage of test watcher from the base class.
 */
@RunWith(JUnit4.class)
public abstract class TimeoutTestWatcherBaseTest {
  boolean timeoutCaught = false;

  @Rule
  public TimeoutTestWatcher testWatcher =
      new TimeoutTestWatcher() {
        @Override
        protected long getTimeoutMillis() {
          return 100;
        }

        @Override
        protected boolean onTimeout() {
          return timeoutCaught = true;
        }
      };
}
