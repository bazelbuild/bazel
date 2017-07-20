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

package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.RegisteredToolchainsFunction.InvalidTargetException;
import com.google.devtools.build.lib.skyframe.ToolchainResolutionValue.ToolchainResolutionKey;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** {@link SkyFunction} which performs toolchain resolution for a class of rules. */
public class ToolchainResolutionFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    ToolchainResolutionKey key = (ToolchainResolutionKey) skyKey.argument();

    BuildConfiguration configuration = key.configuration();
    PlatformConfiguration platformConfiguration =
        configuration.getFragment(PlatformConfiguration.class);

    if (platformConfiguration.hasToolchainOverride(key.toolchainType())) {
      // Short circuit everything and just return the override.
      return ToolchainResolutionValue.create(
          platformConfiguration.getToolchainOverride(key.toolchainType()));
    }

    // Get all toolchains.
    RegisteredToolchainsValue toolchains;
    try {
      toolchains =
          (RegisteredToolchainsValue)
              env.getValueOrThrow(
                  RegisteredToolchainsValue.key(key.configuration()),
                  ConfiguredValueCreationException.class,
                  InvalidTargetException.class,
                  EvalException.class);
      if (toolchains == null) {
        return null;
      }
    } catch (ConfiguredValueCreationException e) {
      throw new ToolchainResolutionFunctionException(e);
    } catch (InvalidTargetException e) {
      throw new ToolchainResolutionFunctionException(e);
    } catch (EvalException e) {
      throw new ToolchainResolutionFunctionException(e);
    }

    // Find the right one.
    DeclaredToolchainInfo toolchain =
        resolveConstraints(
            key.toolchainType(),
            key.execPlatform(),
            key.targetPlatform(),
            toolchains.registeredToolchains());

    if (toolchain == null) {
      throw new ToolchainResolutionFunctionException(
          new NoToolchainFoundException(key.toolchainType()));
    }
    return ToolchainResolutionValue.create(toolchain.toolchainLabel());
  }

  @VisibleForTesting
  static DeclaredToolchainInfo resolveConstraints(
      Label toolchainType,
      PlatformInfo execPlatform,
      PlatformInfo targetPlatform,
      ImmutableList<DeclaredToolchainInfo> toolchains) {
    for (DeclaredToolchainInfo toolchain : toolchains) {
      // Make sure the type matches.
      if (!toolchain.toolchainType().equals(toolchainType)) {
        continue;
      }
      if (!checkConstraints(toolchain.execConstraints(), execPlatform)) {
        continue;
      }
      if (!checkConstraints(toolchain.targetConstraints(), targetPlatform)) {
        continue;
      }

      return toolchain;
    }

    return null;
  }

  /**
   * Returns {@code true} iff all constraints set by the toolchain are present in the {@link
   * PlatformInfo}.
   */
  private static boolean checkConstraints(
      Iterable<ConstraintValueInfo> toolchainConstraints, PlatformInfo platform) {

    for (ConstraintValueInfo constraint : toolchainConstraints) {
      ConstraintValueInfo found = platform.getConstraint(constraint.constraint());
      if (!constraint.equals(found)) {
        return false;
      }
    }
    return true;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** Used to indicate that a toolchain was not found for the current request. */
  public static final class NoToolchainFoundException extends NoSuchThingException {
    private final Label missingToolchainType;

    public NoToolchainFoundException(Label missingToolchainType) {
      super(String.format("no matching toolchain found for %s", missingToolchainType));
      this.missingToolchainType = missingToolchainType;
    }

    public Label missingToolchainType() {
      return missingToolchainType;
    }
  }

  /** Used to indicate errors during the computation of an {@link ToolchainResolutionValue}. */
  private static final class ToolchainResolutionFunctionException extends SkyFunctionException {
    public ToolchainResolutionFunctionException(NoToolchainFoundException e) {
      super(e, Transience.PERSISTENT);
    }

    public ToolchainResolutionFunctionException(ConfiguredValueCreationException e) {
      super(e, Transience.PERSISTENT);
    }

    public ToolchainResolutionFunctionException(InvalidTargetException e) {
      super(e, Transience.PERSISTENT);
    }

    public ToolchainResolutionFunctionException(EvalException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
