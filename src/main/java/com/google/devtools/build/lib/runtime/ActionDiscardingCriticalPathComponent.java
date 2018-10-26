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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.cmdline.Label;
import javax.annotation.Nullable;

/**
 * {@link CriticalPathComponent} that clears its {@link Action} reference after the component
 * is finished running. This allows the action to be GC'ed, if other Bazel components are configured
 * to also release their references to the action.
 *
 * <p>This class is separate from {@link CriticalPathComponent} for memory reasons: the
 * additional references here add between 8 and 12 bytes per component instance with compressed OOPS
 * and 24 bytes without compressed OOPS, which is a price we'd rather not pay unless the user
 * explicitly enables this clearing of {@link Action} objects.
 */
class ActionDiscardingCriticalPathComponent extends CriticalPathComponent {
  @Nullable private final Label owner;
  private final String prettyPrint;
  private final String mnemonic;

  ActionDiscardingCriticalPathComponent(int id, Action action, long relativeStartNanos) {
    super(id, action, relativeStartNanos);
    this.prettyPrint = super.prettyPrintAction();
    this.owner = super.getOwner();
    this.mnemonic = super.getMnemonic();
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  public String prettyPrintAction() {
    return prettyPrint;
  }

  @Nullable
  @Override
  public Label getOwner() {
    return owner;
  }

  @Override
  public synchronized void finishActionExecution(long startNanos, long finishNanos) {
    super.finishActionExecution(startNanos, finishNanos);
    this.action = null;
  }
}
