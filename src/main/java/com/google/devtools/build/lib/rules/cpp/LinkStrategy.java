// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

/**
 * A strategy for executing {@link CppLinkAction}s.
 *
 * <p>The linker commands, e.g. "ar", are not necessary functional, i.e.
 * they may mutate the output file rather than overwriting it.
 * To avoid this, we need to delete the output file before invoking the
 * command.  That must be done by the classes that extend this class.
 */
public abstract class LinkStrategy implements CppLinkActionContext {
  public LinkStrategy() {
  }

  /** The strategy name, preferably suitable for passing to --link_strategy. */
  public abstract String linkStrategyName();

  @Override
  public String strategyLocality(CppLinkAction execOwner) {
    return linkStrategyName();
  }
}
