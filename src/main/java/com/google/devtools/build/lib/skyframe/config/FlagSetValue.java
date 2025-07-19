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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.base.Strings.nullToEmpty;
import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.ConfigFlagDefinitions;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** A return value of {@link FlagSetFunction} */
public class FlagSetValue implements SkyValue {

  private final ImmutableSet<String> flags;

  /**
   * Warnings and info messages for the caller to emit. This lets the caller persistently emit
   * messages that Skyframe ignores on cache hits. See {@link Reportable#storeForReplay}).
   */
  private final ImmutableSet<Event> persistentMessages;

  /** Key for {@link FlagSetValue} based on the raw flags. */
  @ThreadSafety.Immutable
  @AutoCodec
  public record Key(
      ImmutableSet<Label> targets,
      Label projectFile,
      @Nullable String sclConfig,
      BuildOptions targetOptions,
      ImmutableMap<String, String> userOptions,
      ConfigFlagDefinitions configFlagDefinitions,
      boolean enforceCanonical)
      implements SkyKey {

    public Key {
      requireNonNull(targets, "targets");
      requireNonNull(projectFile, "projectFile");
      sclConfig = nullToEmpty(sclConfig);
      requireNonNull(targetOptions, "targetOptions");
      requireNonNull(userOptions, "userOptions");
      requireNonNull(configFlagDefinitions, "configFlagDefinitions");
    }

    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    /**
     * Creating @link FlagSetValue.Key. b/409382048 requires to pass the targets to the Key so it
     * can be used in FlagSetFunction. But this is bad for Skyframe caching. For the sake of fast
     * iteration, this is the simplest approach. We should consider to optimize this in the future.
     */
    public static Key create(
        ImmutableSet<Label> targets,
        Label projectFile,
        String sclConfig,
        BuildOptions targetOptions,
        ImmutableMap<String, String> userOptions,
        ConfigFlagDefinitions configFlagDefinitions,
        boolean enforceCanonical) {
      return interner.intern(
          new Key(
              targets,
              projectFile,
              sclConfig,
              targetOptions,
              userOptions,
              configFlagDefinitions,
              enforceCanonical));
    }

    @Override
    public SkyKeyInterner<?> getSkyKeyInterner() {
      return interner;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.FLAG_SET;
    }
  }

  public static FlagSetValue create(
      ImmutableSet<String> flags, ImmutableSet<Event> persistentMessages) {
    return new FlagSetValue(flags, persistentMessages);
  }

  public FlagSetValue(ImmutableSet<String> flags, ImmutableSet<Event> persistentMessages) {
    this.flags = flags;
    this.persistentMessages = persistentMessages;
  }

  /** Returns the set of flags to be applied to the build from the flagset, in flag=value form. */
  public ImmutableSet<String> getOptionsFromFlagset() {
    return flags;
  }

  public ImmutableSet<Event> getPersistentMessages() {
    return persistentMessages;
  }
}
