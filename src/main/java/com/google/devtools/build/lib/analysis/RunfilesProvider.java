// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import com.google.common.base.Function;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;

/**
 * Runfiles a target contributes to targets that depend on it.
 *
 * <p>The set of runfiles contributed can be different if the dependency is through a <code>data
 * </code> attribute (note that this is just a rough approximation of the reality -- rule
 * implementations are free to request the data runfiles at any time)
 */
@Immutable
@AutoCodec
public final class RunfilesProvider implements TransitiveInfoProvider {
  private final Runfiles defaultRunfiles;
  private final Runfiles dataRunfiles;

  @VisibleForSerialization
  RunfilesProvider(Runfiles defaultRunfiles, Runfiles dataRunfiles) {
    this.defaultRunfiles = defaultRunfiles;
    this.dataRunfiles = dataRunfiles;
  }

  public Runfiles getDefaultRunfiles() {
    return defaultRunfiles;
  }

  public Runfiles getDataRunfiles() {
    return dataRunfiles;
  }

  /**
   * Returns a function that gets the default runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> DEFAULT_RUNFILES =
      new Function<TransitiveInfoCollection, Runfiles>() {
        @Override
        public Runfiles apply(TransitiveInfoCollection input) {
          RunfilesProvider provider = input.getProvider(RunfilesProvider.class);
          if (provider != null) {
            return provider.getDefaultRunfiles();
          }

          return Runfiles.EMPTY;
        }
      };

  /**
   * Returns a function that gets the data runfiles from a {@link TransitiveInfoCollection} or the
   * empty runfiles instance if it does not contain that provider.
   *
   * <p>These are usually used if the target is depended on through a {@code data} attribute.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> DATA_RUNFILES =
      new Function<TransitiveInfoCollection, Runfiles>() {
        @Override
        public Runfiles apply(TransitiveInfoCollection input) {
          RunfilesProvider provider = input.getProvider(RunfilesProvider.class);
          if (provider != null) {
            return provider.getDataRunfiles();
          }

          return Runfiles.EMPTY;
        }
      };

  public static RunfilesProvider simple(Runfiles defaultRunfiles) {
    return new RunfilesProvider(defaultRunfiles, defaultRunfiles);
  }

  public static RunfilesProvider withData(
      Runfiles defaultRunfiles, Runfiles dataRunfiles) {
    return new RunfilesProvider(defaultRunfiles, dataRunfiles);
  }

  public static final RunfilesProvider EMPTY = new RunfilesProvider(
      Runfiles.EMPTY, Runfiles.EMPTY);
}
