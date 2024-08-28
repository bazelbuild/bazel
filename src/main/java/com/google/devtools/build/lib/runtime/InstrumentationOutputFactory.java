// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Creates different types of {@link InstrumentationOutputBuilder}. */
public final class InstrumentationOutputFactory {
  private final Supplier<LocalInstrumentationOutput.Builder>
      localInstrumentationOutputBuilderSupplier;

  private final Supplier<BuildEventArtifactInstrumentationOutput.Builder>
      buildEventArtifactInstrumentationOutputBuilderSupplier;

  private InstrumentationOutputFactory(
      Supplier<LocalInstrumentationOutput.Builder> localInstrumentationOutputBuilderSupplier,
      Supplier<BuildEventArtifactInstrumentationOutput.Builder>
          buildEventArtifactInstrumentationOutputBuilderSupplier) {
    this.localInstrumentationOutputBuilderSupplier = localInstrumentationOutputBuilderSupplier;
    this.buildEventArtifactInstrumentationOutputBuilderSupplier =
        buildEventArtifactInstrumentationOutputBuilderSupplier;
  }

  public LocalInstrumentationOutput.Builder createLocalInstrumentationOutputBuilder() {
    return localInstrumentationOutputBuilderSupplier.get();
  }

  public BuildEventArtifactInstrumentationOutput.Builder
      createBuildEventArtifactInstrumentationOutputBuilder() {
    return buildEventArtifactInstrumentationOutputBuilderSupplier.get();
  }

  /** Builder for {@link InstrumentationOutputFactory}. */
  public static class Builder {
    @Nullable
    private Supplier<LocalInstrumentationOutput.Builder> localInstrumentationOutputBuilderSupplier;

    @Nullable
    private Supplier<BuildEventArtifactInstrumentationOutput.Builder>
        buildEventArtifactInstrumentationOutputBuilderSupplier;

    @CanIgnoreReturnValue
    public Builder setLocalInstrumentationOutputBuilderSupplier(
        Supplier<LocalInstrumentationOutput.Builder> localInstrumentationOutputBuilderSupplier) {
      this.localInstrumentationOutputBuilderSupplier = localInstrumentationOutputBuilderSupplier;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setBuildEventArtifactInstrumentationOutputBuilderSupplier(
        Supplier<BuildEventArtifactInstrumentationOutput.Builder>
            buildEventArtifactInstrumentationOutputBuilderSupplier) {
      this.buildEventArtifactInstrumentationOutputBuilderSupplier =
          buildEventArtifactInstrumentationOutputBuilderSupplier;
      return this;
    }

    public InstrumentationOutputFactory build() {
      return new InstrumentationOutputFactory(
          checkNotNull(
              localInstrumentationOutputBuilderSupplier,
              "Cannot create InstrumentationOutputFactory without localOutputBuilderSupplier"),
          checkNotNull(
              buildEventArtifactInstrumentationOutputBuilderSupplier,
              "Cannot create InstrumentationOutputFactory without bepOutputBuilderSupplier"));
    }
  }
}
