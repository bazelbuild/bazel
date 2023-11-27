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

import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * After resolving bazel module this event is sent from {@link BazelDepGraphFunction} holding the
 * lockfile value with module updates, and the module extension usages. It will be received in
 * {@link BazelLockFileModule} to be used to update the lockfile content
 *
 * <p>Instances of this class are retained in Skyframe nodes and subject to frequent {@link
 * java.util.Set}-based deduplication. As such, it <b>must</b> have a cheap implementation of {@link
 * #hashCode()} and {@link #equals(Object)}. It currently uses reference equality since the logic
 * that creates instances of this class already ensures that there is only one instance per build.
 */
public final class BazelModuleResolutionEvent implements Postable {

  private final BazelLockFileValue onDiskLockfileValue;
  private final BazelLockFileValue resolutionOnlyLockfileValue;
  private final ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
      extensionUsagesById;

  private BazelModuleResolutionEvent(
      BazelLockFileValue onDiskLockfileValue,
      BazelLockFileValue resolutionOnlyLockfileValue,
      ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesById) {
    this.onDiskLockfileValue = onDiskLockfileValue;
    this.resolutionOnlyLockfileValue = resolutionOnlyLockfileValue;
    this.extensionUsagesById = extensionUsagesById;
  }

  public static BazelModuleResolutionEvent create(
      BazelLockFileValue onDiskLockfileValue,
      BazelLockFileValue resolutionOnlyLockfileValue,
      ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesById) {
    return new BazelModuleResolutionEvent(
        onDiskLockfileValue, resolutionOnlyLockfileValue, extensionUsagesById);
  }

  /**
   * Returns the contents of the lockfile as it existed on disk before the current build.
   */
  public BazelLockFileValue getOnDiskLockfileValue() {
    return onDiskLockfileValue;
  }

  /**
   * Returns the result of Bazel module resolution in the form of a lockfile without any
   * information about module extension results.
   */
  public BazelLockFileValue getResolutionOnlyLockfileValue() {
    return resolutionOnlyLockfileValue;
  }

  public ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
      getExtensionUsagesById() {
    return extensionUsagesById;
  }

  @Override
  public boolean storeForReplay() {
    return true;
  }
}
