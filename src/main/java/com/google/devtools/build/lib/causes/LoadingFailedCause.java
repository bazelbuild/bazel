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
package com.google.devtools.build.lib.causes;

import com.google.devtools.build.lib.cmdline.Label;

/**
 * Failure due to something associated with a label; also adds a message. The difference between
 * this class and {@link LabelCause} is that instances of this class get posted to the EventBus as
 * {@link com.google.devtools.build.lib.pkgcache.LoadingFailureEvent}.
 */
public class LoadingFailedCause extends LabelCause {
  public LoadingFailedCause(Label label, String msg) {
    super(label, msg);
  }
}
