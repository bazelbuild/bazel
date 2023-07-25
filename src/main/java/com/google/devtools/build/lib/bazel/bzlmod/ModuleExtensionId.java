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

import static com.google.common.collect.Comparators.emptiesFirst;
import static java.util.Comparator.comparing;

import com.google.auto.value.AutoValue;
import com.google.common.base.Splitter;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/** A unique identifier for a {@link ModuleExtension}. */
@AutoValue
public abstract class ModuleExtensionId {
  public static final Comparator<ModuleExtensionId> LEXICOGRAPHIC_COMPARATOR =
      comparing(ModuleExtensionId::getBzlFileLabel)
          .thenComparing(ModuleExtensionId::getExtensionName)
          .thenComparing(
              ModuleExtensionId::getIsolationKey,
              emptiesFirst(IsolationKey.LEXICOGRAPHIC_COMPARATOR));

  /** A unique identifier for a single isolated usage of a fixed module extension. */
  @AutoValue
  abstract static class IsolationKey {
    static final Comparator<IsolationKey> LEXICOGRAPHIC_COMPARATOR =
        comparing(IsolationKey::getModule, ModuleKey.LEXICOGRAPHIC_COMPARATOR)
            .thenComparing(IsolationKey::getUsageExportedName);

    /** The module which contains this isolated usage of a module extension. */
    public abstract ModuleKey getModule();

    /** The exported name of the first extension proxy for this usage. */
    public abstract String getUsageExportedName();

    public static IsolationKey create(ModuleKey module, String usageExportedName) {
      return new AutoValue_ModuleExtensionId_IsolationKey(module, usageExportedName);
    }

    @Override
    public final String toString() {
      return getModule() + "~" + getUsageExportedName();
    }

    public static IsolationKey fromString(String s) throws Version.ParseException {
      List<String> isolationKeyParts = Splitter.on("~").splitToList(s);
      return ModuleExtensionId.IsolationKey.create(
          ModuleKey.fromString(isolationKeyParts.get(0)), isolationKeyParts.get(1));
    }
  }

  public abstract Label getBzlFileLabel();

  public abstract String getExtensionName();

  public abstract Optional<IsolationKey> getIsolationKey();

  public static ModuleExtensionId create(
      Label bzlFileLabel, String extensionName, Optional<IsolationKey> isolationKey) {
    return new AutoValue_ModuleExtensionId(bzlFileLabel, extensionName, isolationKey);
  }

  public String asTargetString() {
    return String.format(
        "%s%%%s", getBzlFileLabel().getUnambiguousCanonicalForm(), getExtensionName());
  }
}
