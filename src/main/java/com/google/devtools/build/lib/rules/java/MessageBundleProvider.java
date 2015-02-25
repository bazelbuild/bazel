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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Marks configured targets that are able to supply message bundles to their
 * dependents.
 */
@Immutable
public final class MessageBundleProvider implements TransitiveInfoProvider {

  private final ImmutableList<Artifact> messages;

  public MessageBundleProvider(ImmutableList<Artifact> messages) {
    this.messages = messages;
  }

  /**
   * The set of XML source files containing the message definitions.
   */
  public ImmutableList<Artifact> getMessages() {
    return messages;
  }
}
