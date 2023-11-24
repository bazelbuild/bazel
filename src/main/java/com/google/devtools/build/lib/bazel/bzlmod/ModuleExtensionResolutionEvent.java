// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * After evaluating any module extension this event is sent from {@link SingleExtensionEvalFunction}
 * holding the extension id and the resolution data LockFileModuleExtension. It will be received in
 * {@link BazelLockFileModule} to be used to update the lockfile content.
 *
 * <p>Instances of this class are retained in Skyframe nodes and subject to frequent {@link
 * java.util.Set}-based deduplication. As such, it <b>must</b> have a cheap implementation of {@link
 * #hashCode()} and {@link #equals(Object)}. It currently uses reference equality since the logic
 * that creates instances of this class already ensures that there is only one instance per
 * extension id.
 */
public final class ModuleExtensionResolutionEvent implements Postable {

  private final ModuleExtensionId extensionId;
  private final ModuleExtensionEvalFactors extensionFactors;
  private final LockFileModuleExtension moduleExtension;

  private ModuleExtensionResolutionEvent(
      ModuleExtensionId extensionId,
      ModuleExtensionEvalFactors extensionFactors,
      LockFileModuleExtension moduleExtension) {
    this.extensionId = extensionId;
    this.extensionFactors = extensionFactors;
    this.moduleExtension = moduleExtension;
  }

  public static ModuleExtensionResolutionEvent create(
      ModuleExtensionId extensionId,
      ModuleExtensionEvalFactors extensionFactors,
      LockFileModuleExtension lockfileModuleExtension) {
    return new ModuleExtensionResolutionEvent(
        extensionId, extensionFactors, lockfileModuleExtension);
  }

  public ModuleExtensionId getExtensionId() {
    return extensionId;
  }

  public ModuleExtensionEvalFactors getExtensionFactors() {
    return extensionFactors;
  }

  public LockFileModuleExtension getModuleExtension() {
    return moduleExtension;
  }

  @Override
  public boolean storeForReplay() {
    return true;
  }
}
