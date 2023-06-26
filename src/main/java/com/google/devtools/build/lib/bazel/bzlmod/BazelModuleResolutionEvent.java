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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * After resolving bazel module this event is sent from {@link BazelDepGraphFunction} holding the
 * lockfile value with module updates, and the module extension usages. It will be received in
 * {@link BazelLockFileModule} to be used to update the lockfile content
 */
@AutoValue
public abstract class BazelModuleResolutionEvent implements Postable {

  public static BazelModuleResolutionEvent create(
      BazelLockFileValue lockFileValue,
      ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage> extensionUsagesById) {
    return new AutoValue_BazelModuleResolutionEvent(lockFileValue, extensionUsagesById);
  }

  public abstract BazelLockFileValue getLockfileValue();

  public abstract ImmutableTable<ModuleExtensionId, ModuleKey, ModuleExtensionUsage>
      getExtensionUsagesById();
}
