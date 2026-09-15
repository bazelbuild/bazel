// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;

/** An interface for Exceptions that will contain a {@link DetailedExitCode} */
public interface DetailedException {
  DetailedExitCode getDetailedExitCode();

  @Nullable
  static DetailedExitCode getDetailedExitCode(Exception exception) {
    return exception instanceof DetailedException detailedException
        ? detailedException.getDetailedExitCode()
        : null;
  }
}
