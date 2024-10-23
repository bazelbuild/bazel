// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.common.options;

import java.util.function.Predicate;

/** Utility functions for global RC files. */
final class GlobalRcUtils {

  private GlobalRcUtils() {}

  /** No global RC files in Bazel. Consider "client" options to be global. */
  static final Predicate<ParsedOptionDescription> IS_GLOBAL_RC_OPTION =
      // LINT.IfChange
      (option) -> {
        if (option.getOrigin().getSource() != null
            && option.getOrigin().getSource().equals("client")) {
          return true;
        }
        if (option.getOrigin().getSource() != null
            && option.getOrigin().getSource().equals("Invocation policy")) {
          return true;
        }
        return false;
      };
  // LINT.ThenChange(//src/main/cpp/option_processor.cc,
  // src/main/java/com/google/devtools/common/options/InvocationPolicyEnforcer.java
}
