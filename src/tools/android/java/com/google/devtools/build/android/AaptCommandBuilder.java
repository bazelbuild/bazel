// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;

import com.android.builder.core.VariantConfiguration.Type;
import com.android.sdklib.repository.FullRevision;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import javax.annotation.Nullable;

class AaptCommandBuilder {
  private final Path aapt;
  private final String command;
  private final List<String> flags = new ArrayList<>();
  private final FullRevision buildToolsVersion;
  private final Type variantType;

  AaptCommandBuilder(
      Path aapt, @Nullable FullRevision buildToolsVersion, Type variantType, String command) {
    this.aapt = aapt;
    this.buildToolsVersion = buildToolsVersion;
    this.variantType = variantType;
    this.command = command;
  }

  AaptCommandBuilder add(String flag) {
    flags.add(flag);
    return this;
  }

  AaptCommandBuilder add(String flag, @Nullable String value) {
    if (!Strings.isNullOrEmpty(value)) {
      flags.add(flag);
      flags.add(value);
    }
    return this;
  }

  AaptCommandBuilder add(String flag, @Nullable Path path) {
    if (path != null) {
      add(flag, path.toString());
    }
    return this;
  }

  AaptCommandBuilder addRepeated(String flag, Collection<String> values) {
    for (String value : values) {
      add(flag, value);
    }
    return this;
  }


  AaptCommandBuilder maybeAdd(String flag, boolean condition) {
    if (condition) {
      add(flag);
    }
    return this;
  }

  AaptCommandBuilder maybeAdd(String flag, Path directory, boolean condition) {
    if (condition) {
      add(flag, directory);
    }
    return this;
  }

  AaptCommandBuilder maybeAdd(String flag, FullRevision requiredVersion) {
    if (buildToolsVersion == null || buildToolsVersion.compareTo(requiredVersion) >= 0) {
      add(flag);
    }
    return this;
  }

  AaptCommandBuilder maybeAdd(String flag, Type variant) {
    if (variantType == variant) {
      add(flag);
    }
    return this;
  }

  AaptCommandBuilder maybeAdd(String flag, @Nullable String value, Type variant) {
    if (variantType == variant) {
      add(flag, value);
    }
    return this;
  }

  List<String> build() {
    return ImmutableList
        .<String>builder()
        .add(aapt.toString())
        .add(command)
        .addAll(flags)
        .build();
  }
}

