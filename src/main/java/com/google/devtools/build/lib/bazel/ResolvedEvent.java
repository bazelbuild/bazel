// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel;

import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.XattrProvider;

/** Interface for events reporting information to be added to a resolved file. */
public interface ResolvedEvent extends ExtendedEventHandler.Postable {
  /** The name of the resolved entity, e.g., the name of an external repository. */
  String getName();

  /** The entry for the list of resolved Information. */
  Object getResolvedInformation(XattrProvider xattrProvider);
}
