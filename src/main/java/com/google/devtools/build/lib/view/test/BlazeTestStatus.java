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

package com.google.devtools.build.lib.view.test;

import com.google.testing.proto.TestStatus;
import com.google.testing.proto.TestTargetResult;

/**
 * Details the status of a test run by Blaze
 */
public enum BlazeTestStatus {
    NO_STATUS, PASSED, FLAKY, TIMEOUT, FAILED, INCOMPLETE, REMOTE_FAILURE,
    FAILED_TO_BUILD;

  public BlazeTestStatus aggregateStatus(BlazeTestStatus status) {
    return status.ordinal() > ordinal() ? status : this;
  }

  /**
   * @return true iff status represents successful test.
   */
  public boolean isPassed() {
    return this == PASSED || this == FLAKY;
  }

  /**
   * Returns key used to sort entries on summary. PASSED entries will be
   * displayed first followed by everything else.
   */
  public int getSortKey() {
    return (this == PASSED) ? -1 : ordinal();
  }

  @Override
  public String toString() { return name().replace('_', ' '); }

  /**
   * Maps values from enum TestStatus (used by the
   * testing_api.proto) to this enum.
   *
   * TODO(bazel-team): (2010) This should be restructured in better fashion - perhaps
   * as an adapter class.
   */
  public static BlazeTestStatus getStatusFromTestTargetResult(TestTargetResult result) {
    TestStatus status = result.getStatus();
    if (status == TestStatus.PASSED) {
      return result.getAttemptsCount() > 0 ? FLAKY : PASSED;
    } else if (status == TestStatus.FAILED) {
      return FAILED;
    } else if (status == TestStatus.TIMEOUT) {
      return TIMEOUT;
    } else if (status == TestStatus.FAILED_TO_BUILD) {
      return FAILED_TO_BUILD;
    } else if (status == TestStatus.INTERRUPTED) {
      return INCOMPLETE;
    } else if (status == TestStatus.INTERNAL_ERROR) {
      return REMOTE_FAILURE;
    } else {
      return NO_STATUS;
    }
  }
}
