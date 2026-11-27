// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.google.devtools.build.lib.actions.CommandLineItem;

/**
 * A {@link CommandLineItem} that expands to the path string of a {@link PathFragment}.
 *
 * <p>This allows consumers that only need the bytes of the expansion, such as fingerprinting, to
 * avoid allocating the path string.
 */
public interface PathCommandLineItem extends CommandLineItem {
  /**
   * Returns the path fragment whose {@link PathFragment#getPathString()} equals {@link
   * #expandToCommandLine()}.
   */
  PathFragment getCommandLinePath();
}
