// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Optional;

/** A unique identifier for a {@link ModuleExtension}. */
@AutoValue
public abstract class ModuleExtensionId {

  /** A unique identifier for a single isolated usage of a fixed module extension. */
  @AutoValue
  abstract static class IsolationKey {
    /** The module which contains this isolated usage of a module extension. */
    public abstract ModuleKey getModule();

    /** Whether this isolated usage specified {@code dev_dependency = True}. */
    public abstract boolean isDevUsage();

    /**
     * The 0-based index of this isolated usage within the module's isolated usages of the same
     * module extension and with the same {@link #isDevUsage()} value.
     */
    public abstract int getIsolatedUsageIndex();

    public static IsolationKey create(
        ModuleKey module, boolean isDevUsage, int isolatedUsageIndex) {
      return new AutoValue_ModuleExtensionId_IsolationKey(module, isDevUsage, isolatedUsageIndex);
    }
  }

  public abstract Label getBzlFileLabel();

  public abstract String getExtensionName();

  public abstract Optional<IsolationKey> getIsolationKey();

  public static ModuleExtensionId create(
      Label bzlFileLabel, String extensionName, Optional<IsolationKey> isolationKey) {
    return new AutoValue_ModuleExtensionId(bzlFileLabel, extensionName, isolationKey);
  }
}
