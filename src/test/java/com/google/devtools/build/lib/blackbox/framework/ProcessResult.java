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

package com.google.devtools.build.lib.blackbox.framework;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.util.StringUtilities;
import java.util.List;

/** Result of the external process execution, see {@link ProcessRunner} */
@AutoValue
public abstract class ProcessResult {

  static ProcessResult create(int exitCode, List<String> out, List<String> err) {
    return new AutoValue_ProcessResult(exitCode, out, err);
  }

  abstract int exitCode();

  abstract List<String> out();

  abstract List<String> err();

  public String outString() {
    return StringUtilities.joinLines(out());
  }

  public String errString() {
    return StringUtilities.joinLines(err());
  }
}
