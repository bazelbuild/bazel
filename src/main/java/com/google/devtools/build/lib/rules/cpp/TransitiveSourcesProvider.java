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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Provides information on sources in the transitive closure of a target. */
// TODO(b/65016770): Add SWIFT to this provider.
public class TransitiveSourcesProvider implements TransitiveInfoProvider {

  /** Signal that a certain source type is used. */
  public enum SourceUsage {
    CPP(FileTypeSet.of(CppFileTypes.CPP_SOURCE, CppFileTypes.OBJCPP_SOURCE)),
    OBJC(FileTypeSet.of(CppFileTypes.OBJC_SOURCE));

    private final FileTypeSet fileType;

    SourceUsage(FileTypeSet fileType) {
      this.fileType = fileType;
    }

    /** If true, the presence of the given source signals this source usage. */
    public boolean matches(String fileName) {
      return fileType.matches(fileName);
    }
  }

  private final ImmutableList<SourceUsage> sources;

  private TransitiveSourcesProvider(ImmutableList<SourceUsage> sources) {
    this.sources = sources;
  }

  /** True if sources of the given type are used in this build. */
  public boolean uses(SourceUsage source) {
    return sources.contains(source);
  }

  /** Builder for TransitiveSourcesProvider */
  public static class Builder {
    private final ImmutableList.Builder<SourceUsage> sources = ImmutableList.builder();

    /** Signals that the build uses sources of the provided type. */
    public Builder doesUse(SourceUsage source) {
      this.sources.add(source);
      return this;
    }

    public TransitiveSourcesProvider build() {
      return new TransitiveSourcesProvider(this.sources.build());
    }
  }
}
