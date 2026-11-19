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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.Forward;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.ForwardKey;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.Inverse;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.InverseKey;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.ModuleInfo;
import com.google.devtools.build.lib.skyframe.RepositoryMappingEntryValue.ModuleInfoKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Optional;
import javax.annotation.Nullable;

/** {@link SkyFunction} for {@link RepositoryMappingEntryValue}s. */
public final class RepositoryMappingEntryFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
    var key = (RepositoryMappingEntryValue.Key) skyKey;
    var mappingValue =
        (RepositoryMappingValue) env.getValue(RepositoryMappingValue.key(key.contextRepo()));
    if (mappingValue == null) {
      return null;
    }
    RepositoryMapping mapping = mappingValue.repositoryMapping();
    return switch (key) {
      case ForwardKey forwardKey ->
          mapping == null
              ? Forward.REPO_NOT_FOUND
              : Forward.of(mapping.get(forwardKey.apparentName()));
      case InverseKey inverseKey ->
          Inverse.of(
              mapping == null ? Optional.empty() : mapping.getInverse(inverseKey.canonicalName()));
      case ModuleInfoKey unused ->
          mapping == null
              ? ModuleInfo.REPO_NOT_FOUND
              : new ModuleInfo(
                  true,
                  mappingValue.associatedModuleName(),
                  mappingValue.associatedModuleVersion(),
                  mappingValue.moduleRepoName());
    };
  }
}
