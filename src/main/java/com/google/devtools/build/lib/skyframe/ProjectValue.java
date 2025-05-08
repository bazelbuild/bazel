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
package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.SimpleTargetPatternMatcher;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.Objects;
import javax.annotation.Nullable;

/** A SkyValue representing the parsed definitions from a PROJECT.scl file. */
public final class ProjectValue implements SkyValue {
  /**
   * Represents the enforcement policy for a PROJECT.scl file.
   *
   * <p>"warn" (default) - warn if the user set any output-affecting options that are not present in
   * the selected config in a blazerc or on the command line.
   *
   * <p>"compatible" - fail if the user set any options that are present in the selected config to a
   * different value than the one in the config. Also warn for other output-affecting options
   *
   * <p>"strict" - fail if the user set any output-affecting options that are not present in the
   * selected config.
   */
  public enum EnforcementPolicy {
    WARN("warn"), // Default, enforced in ProjectFunction#compute.
    COMPATIBLE("compatible"),
    STRICT("strict");

    EnforcementPolicy(String value) {
      this.value = value;
    }

    private final String value;

    public static EnforcementPolicy fromString(String value) {
      for (EnforcementPolicy policy : EnforcementPolicy.values()) {
        if (policy.value.equals(value)) {
          return policy;
        }
      }
      throw new IllegalArgumentException(String.format("invalid enforcement_policy '%s'", value));
    }
  }

  private final EnforcementPolicy enforcementPolicy;
  private final ImmutableMap<String, Collection<String>> projectDirectories;
  @Nullable private final ImmutableMap<String, BuildableUnit> buildableUnits;
  @Nullable private final ImmutableList<String> alwaysAllowedConfigs;
  @Nullable private final Label actualProjectFile;

  /**
   * A project's buildable units.
   *
   * <p>A buildable unit is a named pair of build flags and target patterns. The name is stored as a
   * map key in {@link ProjectValue#getBuildableUnits()}
   *
   * <p>See {@code third_party/bazel/src/main/protobuf/project/project.proto} for precise
   * definitions.
   */
  @AutoValue
  public abstract static class BuildableUnit {
    /**
     * Creates a buildable unit.
     *
     * @param targetPatterns the buildable unit's target patterns, or empty if they weren't set
     * @param description the buildable unit's user-friendly description, or empty if not set
     * @param flags the buildable unit's flags
     * @param isDefault whether this is the default buildable unit
     */
    public static BuildableUnit create(
        ImmutableList<String> targetPatterns,
        String description,
        ImmutableList<String> flags,
        boolean isDefault)
        throws LabelSyntaxException {
      return new AutoValue_ProjectValue_BuildableUnit(
          SimpleTargetPatternMatcher.create(targetPatterns), description, flags, isDefault);
    }

    public abstract SimpleTargetPatternMatcher targetPatternMatcher();

    public abstract String description();

    public abstract ImmutableList<String> flags();

    public abstract boolean isDefault();
  }

  public ProjectValue(
      EnforcementPolicy enforcementPolicy,
      ImmutableMap<String, Collection<String>> projectDirectories,
      @Nullable ImmutableMap<String, BuildableUnit> buildableUnits,
      @Nullable ImmutableList<String> alwaysAllowedConfigs,
      @Nullable Label actualProjectFile) {
    this.enforcementPolicy = enforcementPolicy;
    this.projectDirectories = projectDirectories;
    this.buildableUnits = buildableUnits;
    this.alwaysAllowedConfigs = alwaysAllowedConfigs;
    this.actualProjectFile = actualProjectFile;
  }

  /**
   * Return the "default" {@code project_directories} map entry. If there are zero entries, returns
   * an empty set.
   */
  public ImmutableSet<String> getDefaultProjectDirectories() {
    if (projectDirectories.isEmpty()) {
      return ImmutableSet.of();
    }
    // TODO: b/409377907 - Make sure this check still makes sense with the new format.
    Preconditions.checkArgument(
        projectDirectories.containsKey("default"),
        "project_directories must contain the 'default' key");
    return ImmutableSet.copyOf(projectDirectories.get("default"));
  }

  /**
   * If a project file has the content
   *
   * {@snippet :
   *   project = {
   *     "actual": "//other:PROJECT.scl"
   *   }
   * }
   *
   * <p>then this is the same project defined canonically in {@code //other:PROJECT.scl} and this
   * method returns {@code //other:PROJECT.scl}. Else returns the {@link ProjectValue.Key} label
   * that produces this value.
   *
   * <p>Files that define "actual" cannot define any other content. That's considered a parsing
   * error.
   */
  public Label getActualProjectFile() {
    return actualProjectFile;
  }

  public EnforcementPolicy getEnforcementPolicy() {
    return enforcementPolicy;
  }

  /**
   * Maps buildable unit names to definitions. Null if not specified. Note that an empty list is not
   * the same as unspecified.
   *
   * <p>Builds can trigger a buildable unit by setting {@code --scl_config=<name>}.
   */
  @Nullable
  public ImmutableMap<String, BuildableUnit> getBuildableUnits() {
    return buildableUnits;
  }

  @Nullable
  public ImmutableList<String> getAlwaysAllowedConfigs() {
    return alwaysAllowedConfigs;
  }

  /**
   * Returns the map of named {@code project_directories} in the project. If the map is not defined
   * in the file, returns an empty map.
   */
  public ImmutableMap<String, Collection<String>> getProjectDirectories() {
    return projectDirectories;
  }

  /** The SkyKey. Uses the label of the project file as the input. */
  public static final class Key implements SkyKey {
    private final Label projectFile;

    public Key(Label projectFile) {
      this.projectFile = Preconditions.checkNotNull(projectFile);
    }

    public Label getProjectFile() {
      return projectFile;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PROJECT;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      ProjectValue.Key key = (ProjectValue.Key) o;
      return Objects.equals(projectFile, key.projectFile);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(projectFile);
    }
  }
}
