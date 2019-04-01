// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildeventstream;

import com.google.devtools.build.lib.util.AbruptExitException;
import java.util.function.Consumer;

/**
 * A callback function that the Build Event Service transports can use to notify of an {@link
 * AbruptExitException} to the main thread which may exit the build abruptly.
 */
// TODO(lpino): Delete this callback once all the transports can depend directly on
//  {@link ModuleEnvironment}.
public interface BuildEventServiceAbruptExitCallback extends Consumer<AbruptExitException> {
  /** Executes the callback using the provided {@link AbruptExitException}. */
  @Override
  void accept(AbruptExitException e);
}
