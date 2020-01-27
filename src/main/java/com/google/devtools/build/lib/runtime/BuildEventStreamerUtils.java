// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;

/**
 * Utility methods for the build event stream.
 *
 * <p>TODO(aehlig): remove once {@link BlazeTestStatus} is replaced by {@link TestStatus} from the
 * {@link BuildEventStreamProtos}.
 */
public final class BuildEventStreamerUtils {

  /** Map BlazeTestStatus to TestStatus. */
  public static TestStatus bepStatus(BlazeTestStatus status) {
    switch (status) {
      case NO_STATUS:
        return BuildEventStreamProtos.TestStatus.NO_STATUS;
      case PASSED:
        return BuildEventStreamProtos.TestStatus.PASSED;
      case FLAKY:
        return BuildEventStreamProtos.TestStatus.FLAKY;
      case FAILED:
        return BuildEventStreamProtos.TestStatus.FAILED;
      case TIMEOUT:
        return BuildEventStreamProtos.TestStatus.TIMEOUT;
      case INCOMPLETE:
        return BuildEventStreamProtos.TestStatus.INCOMPLETE;
      case REMOTE_FAILURE:
        return BuildEventStreamProtos.TestStatus.REMOTE_FAILURE;
      case BLAZE_HALTED_BEFORE_TESTING:
        return BuildEventStreamProtos.TestStatus.TOOL_HALTED_BEFORE_TESTING;
      default:
        // Not used as the above is a complete case distinction; however, by the open
        // nature of protobuf enums, we need the clause to convice java, that we always
        // have a return statement.
        return BuildEventStreamProtos.TestStatus.NO_STATUS;
    }
  }
}
