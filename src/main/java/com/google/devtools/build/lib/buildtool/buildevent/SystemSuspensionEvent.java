// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.devtools.build.lib.events.ExtendedEventHandler;

/**
 * This event is fired from {@code SystemSuspensionModule#suspendCallback} to indicate that the user
 * either suspended the build with a signal or put their computer to sleep.
 */
public class SystemSuspensionEvent implements ExtendedEventHandler.Postable {
  public enum Reason {
    SIGTSTP {
      @Override
      public String logString() {
        return "Signal SIGTSTP";
      }
    },
    SIGCONT {
      @Override
      public String logString() {
        return "Signal SIGCONT";
      }
    },
    SLEEP {
      @Override
      public String logString() {
        return "Computer put to sleep";
      }
    },
    WAKE {
      @Override
      public String logString() {
        return "Computer woken up";
      }
    };

    // These constants are mapped to enum in third_party/bazel/src/main/native/unix_jni.h.
    static Reason fromInt(int number) {
      switch (number) {
        case 0:
          return SIGTSTP;
        case 1:
          return SIGCONT;
        case 2:
          return SLEEP;
        case 3:
          return WAKE;
        default:
          throw new IllegalStateException("Unknown suspension reason: " + number);
      }
    }

    public abstract String logString();
  }

  private final Reason reason;

  public SystemSuspensionEvent(int reason) {
    this.reason = Reason.fromInt(reason);
  }

  public Reason getReason() {
    return reason;
  }

  public String logString() {
    return "SystemSuspensionEvent: " + reason.logString();
  }
}
