// Copyright 2017 The Bazel Authors. All rights reserved.
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

import java.util.Collection;

/** Interface for {@link BuildEvent}s that reference build configurations */
public interface BuildEventWithConfiguration extends BuildEvent {
  /**
   * The configurations the event mentions, and hence should be introduced in the stream before this
   * event; they are abstracted as {@link BuildEvent}, as for the build event stream the only thing
   * we care is how they get presenteded in the protocol.
   */
  Collection<BuildEvent> getConfigurations();
}
