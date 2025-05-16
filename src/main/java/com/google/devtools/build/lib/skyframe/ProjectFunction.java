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

import static com.google.devtools.build.skyframe.SkyFunctionException.Transience.PERSISTENT;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.StarlarkInfoNoSchema;
import com.google.devtools.build.lib.skyframe.ProjectValue.BuildableUnit;
import com.google.devtools.build.lib.skyframe.ProjectValue.EnforcementPolicy;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.StarlarkList;

/** A {@link SkyFunction} that loads metadata from a PROJECT.scl file. */
public class ProjectFunction implements SkyFunction {

  /** The top level reserved globals in the PROJECT.scl file. */
  private enum ReservedGlobals {
    /**
     * Forward-facing PROJECT.scl structure: a single top-level "project" variable that contains all
     * project data in nested data structures.
     */
    PROJECT("project");

    private final String key;

    ReservedGlobals(String key) {
      this.key = key;
    }

    String getKey() {
      return key;
    }
  }

  private static final String ENFORCEMENT_POLICY = "enforcement_policy";

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ProjectFunctionException, InterruptedException {
    ProjectValue.Key key = (ProjectValue.Key) skyKey.argument();

    BzlLoadValue bzlLoadValue;
    try {
      bzlLoadValue =
          (BzlLoadValue)
              env.getValueOrThrow(
                  BzlLoadValue.keyForBuild(key.getProjectFile()), BzlLoadFailedException.class);
    } catch (BzlLoadFailedException e) {
      throw new ProjectFunctionException(e, PERSISTENT);
    }
    if (bzlLoadValue == null) {
      return null;
    }
    Object projectRaw = bzlLoadValue.getModule().getGlobal(ReservedGlobals.PROJECT.getKey());
    switch (projectRaw) {
      case null -> {
        throw new ProjectFunctionException(
            new TypecheckFailureException(
                "Project files must define exactly one top-level variable called \"project\""));
      }
      case Dict<?, ?> asDict -> {
        Label actualProjectFile = maybeResolveAlias(key.getProjectFile(), asDict, bzlLoadValue);
        if (!actualProjectFile.equals(key.getProjectFile())) {
          // This is an alias for another project file. Delegate there.
          // TODO: b/382265245 - handle cycles, including self references.
          return env.getValueOrThrow(
              new ProjectValue.Key(actualProjectFile), ProjectFunctionException.class);
        }
        return parseLegacyProjectSchema(asDict, key.getProjectFile());
      }
      case StarlarkInfoNoSchema starlarkInfo -> {
        return parseProtoProjectSchema(starlarkInfo, key.getProjectFile());
      }
      default ->
          throw new ProjectFunctionException(
              new TypecheckFailureException(
                  String.format(
                      "%s variable: expected a map of string to objects, got %s",
                      ReservedGlobals.PROJECT.getKey(), projectRaw.getClass())));
    }
  }

  /**
   * Parses the proto-based PROJECT.scl implementation.
   *
   * @param starlarkInfo the raw Starlark {@link StarlarkInfoNoSchema} that {@code project} is set
   *     to
   * @param projectFile name of the project file
   */
  private static ProjectValue parseProtoProjectSchema(
      StarlarkInfoNoSchema starlarkInfo, Label projectFile) throws ProjectFunctionException {
    Map<String, BuildableUnit> buildableUnitsBuilder = new LinkedHashMap<>();
    Collection<?> buildableUnits =
        checkAndCast(
            starlarkInfo.getValue("buildable_units"),
            Collection.class,
            /* defaultValue= */ null,
            "buildable_units must be a list of buildable unit definitions");
    for (Object rawBuildableUnit : buildableUnits) {
      ImmutableList.Builder<String> targetPatternsBuilder = ImmutableList.builder();
      ImmutableList.Builder<String> flagsBuilder = ImmutableList.builder();
      StarlarkInfoNoSchema buildableUnitStruct =
          checkAndCast(
              rawBuildableUnit,
              StarlarkInfoNoSchema.class,
              /* defaultValue= */ null,
              "buildable_units entries must be structured objects");
      String buildableUnitName =
          checkAndCast(
              buildableUnitStruct.getValue("name"),
              String.class,
              /* defaultValue= */ null,
              "buildable_unit names must be strings");
      String buildableUnitDescription =
          checkAndCast(
              buildableUnitStruct.getValue("description"),
              String.class,
              /* defaultValue= */ buildableUnitName,
              "buildable_unit descriptions must be strings");
      boolean isDefault =
          checkAndCast(
              buildableUnitStruct.getValue("is_default"),
              Boolean.class,
              /* defaultValue= */ false,
              "is_default must be a boolean");
      Collection<?> targetPatterns =
          checkAndCast(
              buildableUnitStruct.getValue("target_patterns"),
              Collection.class,
              /* defaultValue= */ ImmutableList.of(),
              "target_patterns must be a list of strings");
      for (Object targetPattern : targetPatterns) {
        targetPatternsBuilder.add(
            checkAndCast(
                targetPattern,
                String.class,
                /* defaultValue= */ null,
                "target_patterns entries must be strings"));
      }
      Collection<?> flags =
          checkAndCast(
              buildableUnitStruct.getValue("flags"),
              Collection.class,
              /* defaultValue= */ ImmutableList.of(),
              "flags must be a list of strings");
      for (Object flag : flags) {
        flagsBuilder.add(
            checkAndCast(
                flag, String.class, /* defaultValue= */ null, "flags entries must be strings"));
      }
      // TODO: b/413130912: cleanly fail when multiple buildable units have the same name.
      BuildableUnit buildableUnit = null;
      try {
        buildableUnit =
            BuildableUnit.create(
                targetPatternsBuilder.build(),
                buildableUnitDescription,
                flagsBuilder.build(),
                isDefault);
      } catch (LabelSyntaxException e) {
        throw new ProjectFunctionException(e);
      }
      if (buildableUnitsBuilder.put(buildableUnitName, buildableUnit) != null) {
        throw new ProjectFunctionException(
            new BadProjectFileException(
                String.format(
                    "buildable_unit name='%s' is repeated. Buildable units must have unique names.",
                    buildableUnitName)));
      }
    }
    ImmutableList<String> alwaysAllowedConfigs =
        parseAlwaysAllowedConfigs(starlarkInfo.getValue("always_allowed_configs"));
    return new ProjectValue(
        parseEnforcementPolicy(starlarkInfo.getValue(ENFORCEMENT_POLICY), projectFile),
        parseProjectDirectories(starlarkInfo.getValue("project_directories")),
        ImmutableMap.copyOf(buildableUnitsBuilder),
        alwaysAllowedConfigs.isEmpty() ? null : alwaysAllowedConfigs,
        projectFile);
  }

  /**
   * Parses the first PROJECT.scl implementation (pre-proto schema).
   *
   * @param dict the raw Starlark {@link Dict} that {@code project} is set to
   * @param projectFile name of the project file
   */
  private static ProjectValue parseLegacyProjectSchema(Dict<?, ?> dict, Label projectFile)
      throws ProjectFunctionException {
    ImmutableMap.Builder<String, BuildableUnit> buildableUnitsBuilder = ImmutableMap.builder();
    for (Object k : dict.keySet()) {
      if (!(k instanceof String)) {
        throw new ProjectFunctionException(
            new TypecheckFailureException(
                String.format(
                    "%s variable: expected string key, got element of %s",
                    ReservedGlobals.PROJECT.getKey(), k.getClass())));
      }
    }
    String defaultConfig = null;
    Object defaultConfigRaw = dict.get("default_config");
    if (defaultConfigRaw != null) {
      String defaultConfigString =
          checkAndCast(
              defaultConfigRaw,
              String.class,
              /* defaultValue= */ null,
              "default_config must be a string matching a configs variable definition");
      defaultConfig = defaultConfigString;
    }
    boolean foundDefaultConfig = false;
    if (dict.containsKey("configs")) {
      ImmutableMap<String, Collection<String>> configs =
          parseConfigs(dict.get("configs"), "configs");
      for (String config : configs.keySet()) {
        boolean isDefault = defaultConfig != null && config.equals(defaultConfig);
        if (isDefault) {
          foundDefaultConfig = true;
        }
        BuildableUnit buildableUnit = null;
        try {
          buildableUnit =
              BuildableUnit.create(
                  /* targetPatterns= */ ImmutableList.of(),
                  /* description= */ "",
                  ImmutableList.copyOf(configs.get(config)),
                  isDefault);
        } catch (LabelSyntaxException e) {
          throw new ProjectFunctionException(e);
        }
        buildableUnitsBuilder.put(config, buildableUnit);
      }
    }
    if (defaultConfig != null && !foundDefaultConfig) {
      throw new ProjectFunctionException(
          new BadProjectFileException(
              "default_config must be a string matching a configs variable definition"));
    }
    return new ProjectValue(
        parseEnforcementPolicy(dict.get(ENFORCEMENT_POLICY), projectFile),
        parseProjectDirectories(dict.get("active_directories")),
        dict.containsKey("configs") ? buildableUnitsBuilder.buildOrThrow() : null,
        parseAlwaysAllowedConfigs(dict.get("always_allowed_configs")),
        projectFile);
  }

  private static ImmutableMap<String, Collection<String>> parseConfigs(
      Object configsRaw, String variableName) throws ProjectFunctionException {
    // This project file doesn't define configs, so it must not be used for canonical configs.
    if (configsRaw == null) {
      return ImmutableMap.of();
    }
    ImmutableMap.Builder<String, Collection<String>> configs = ImmutableMap.builder();
    boolean expectedConfigsType = false;
    if (configsRaw instanceof Dict<?, ?> configsAsDict) {
      expectedConfigsType = true;
      for (var entry : configsAsDict.entrySet()) {
        if (!(entry.getKey() instanceof String key
            && entry.getValue() instanceof Collection<?> values)) {
          expectedConfigsType = false;
          break;
        }
        ImmutableList.Builder<String> valuesBuilder = ImmutableList.builder();
        for (var value : values) {
          if (!(value instanceof String val)) {
            expectedConfigsType = false;
            break;
          }
          valuesBuilder.add(val);
        }
        configs.put(key, valuesBuilder.build());
      }
    }
    if (!expectedConfigsType) {
      throw new ProjectFunctionException(
          new TypecheckFailureException(
              String.format(
                  "%s variable must be a map of strings to lists of strings", variableName)));
    }
    return configs.buildOrThrow();
  }

  private static ImmutableList<String> parseAlwaysAllowedConfigs(Object alwaysAllowedConfigsRaw)
      throws ProjectFunctionException {
    if (alwaysAllowedConfigsRaw == null) {
      return ImmutableList.of();
    }
    Collection<?> alwaysAllowedConfigs =
        checkAndCast(
            alwaysAllowedConfigsRaw,
            Collection.class,
            /* defaultValue= */ ImmutableList.of(),
            "always_allowed_configs must be a list of strings");
    ImmutableList.Builder<String> alwaysAllowedConfigsBuilder = ImmutableList.builder();
    for (Object config : alwaysAllowedConfigs) {
      alwaysAllowedConfigsBuilder.add(
          checkAndCast(
              config,
              String.class,
              /* defaultValue= */ null,
              "always_allowed_configs entires must be strings"));
    }
    return alwaysAllowedConfigsBuilder.build();
  }

  private static ImmutableMap<String, Collection<String>> parseProjectDirectories(
      Object activeDirectoriesRaw) throws ProjectFunctionException {
    @SuppressWarnings("unchecked")
    ImmutableMap<String, Collection<String>> activeDirectories =
        switch (activeDirectoriesRaw) {
          case null -> ImmutableMap.of();
          case Dict<?, ?> dict -> {
            ImmutableMap.Builder<String, Collection<String>> builder = ImmutableMap.builder();
            for (Entry<?, ?> entry : dict.entrySet()) {
              Object k = entry.getKey();

              if (!(k instanceof String activeDirectoriesKey)) {
                throw new ProjectFunctionException(
                    new TypecheckFailureException(
                        "expected string, got element of " + k.getClass()));
              }

              Object values = entry.getValue();
              if (!(values instanceof Collection<?> activeDirectoriesValues)) {
                throw new ProjectFunctionException(
                    new TypecheckFailureException(
                        "expected list, got element of " + values.getClass()));
              }

              for (Object activeDirectory : activeDirectoriesValues) {
                if (!(activeDirectory instanceof String)) {
                  throw new ProjectFunctionException(
                      new TypecheckFailureException(
                          "expected a list of strings, got element of "
                              + activeDirectory.getClass()));
                }
              }

              builder.put(activeDirectoriesKey, (Collection<String>) values);
            }

            yield builder.buildOrThrow();
          }
          case List<?> list -> {
            // The proto schema doesn't need a map. Read a list and store as a {"default": [list}]}
            // map to preserve backward compatibility.
            ImmutableList.Builder<String> builder = ImmutableList.builder();
            for (Object activeDirectory : list) {
              builder.add(
                  checkAndCast(
                      activeDirectory,
                      String.class,
                      /* defaultValue= */ null,
                      "project_directories is a list of strings"));
            }
            yield ImmutableMap.of("default", builder.build());
          }
          default ->
              throw new ProjectFunctionException(
                  new TypecheckFailureException(
                      "expected a map of string to list of strings, got "
                          + activeDirectoriesRaw.getClass()));
        };

    if (!activeDirectories.isEmpty() && activeDirectories.get("default") == null) {
      throw new ProjectFunctionException(
          new ActiveDirectoriesException(
              "non-empty active_directories must contain the 'default' key"));
    }
    return activeDirectories;
  }

  private static EnforcementPolicy parseEnforcementPolicy(
      Object enforcementPolicyRaw, Label projectFile) throws ProjectFunctionException {
    if (enforcementPolicyRaw == null
        || ((enforcementPolicyRaw instanceof StarlarkList<?> asList) && asList.isEmpty())) {
      // Default if unspecified.
      return EnforcementPolicy.WARN;
    }
      try {
      return EnforcementPolicy.fromString(enforcementPolicyRaw.toString().toLowerCase(Locale.ROOT));
      } catch (IllegalArgumentException e) {
        throw new ProjectFunctionException(
            new TypecheckFailureException(e.getMessage() + " in " + projectFile));
    }
  }

  /**
   * If this is an alias for another project file, returns its label. Else returns the original
   * key's label.
   *
   * <p>See {@link ProjectValue#maybeResolveAlias} for schema details.
   *
   * @throws ProjectFunctionException if the alias schema isn't valid or the actual reference isn't
   *     a valid label.
   */
  private static Label maybeResolveAlias(
      Label originalProjectFile, Dict<?, ?> project, BzlLoadValue bzlLoadValue)
      throws ProjectFunctionException {
    if (project.get("actual") == null) {
      return originalProjectFile;
    } else if (!(project.get("actual") instanceof String)) {
      throw new ProjectFunctionException(
          new TypecheckFailureException(
              String.format(
                  "project[\"actual\"]: expected string, got %s", project.get("actual"))));
    } else if (project.keySet().size() > 1) {
      throw new ProjectFunctionException(
          new TypecheckFailureException(
              String.format(
                  "project[\"actual\"] is present, but other keys are present as well: %s",
                  project.keySet())));
    } else if (bzlLoadValue.getModule().getGlobals().keySet().size() > 1) {
      throw new ProjectFunctionException(
          new TypecheckFailureException(
              String.format(
                  "project global variable is present, but other globals are present as well: %s",
                  bzlLoadValue.getModule().getGlobals().keySet())));
    }
    try {
      return Label.parseCanonical((String) project.get("actual"));
    } catch (LabelSyntaxException e) {
      throw new ProjectFunctionException(e);
    }
  }

  private static final class TypecheckFailureException extends Exception {
    TypecheckFailureException(String msg) {
      super(msg);
    }
  }

  private static final class ActiveDirectoriesException extends Exception {
    ActiveDirectoriesException(String msg) {
      super(msg);
    }
  }

  private static final class BadProjectFileException extends Exception {
    BadProjectFileException(String msg) {
      super(msg);
    }
  }

  /** Exception thrown by {@link ProjectFunction}. */
  public static final class ProjectFunctionException extends SkyFunctionException {
    ProjectFunctionException(TypecheckFailureException cause) {
      super(cause, PERSISTENT);
    }

    ProjectFunctionException(ActiveDirectoriesException cause) {
      super(cause, PERSISTENT);
    }

    ProjectFunctionException(BzlLoadFailedException e, Transience transience) {
      super(e, transience);
    }

    ProjectFunctionException(LabelSyntaxException cause) {
      super(cause, PERSISTENT);
    }

    ProjectFunctionException(BadProjectFileException cause) {
      super(cause, PERSISTENT);
    }
  }

  /**
   * Checks that {@code rawValue} is an instance of {@code clazz}. If so, returns it cast to that
   * type. Else if its an empty {@link StarlarkList} and {@code defaultValue} is not null, returns
   * {@code defaultValue}. Else throws a {@link ProjectFunctionException}.
   *
   * <p>Note that all unspecified protolark settings default to an empty {@code StarlarkList}.
   */
  private static <T> T checkAndCast(
      Object rawValue, Class<T> clazz, @Nullable Object defaultValue, String errorMessage)
      throws ProjectFunctionException {
    if (clazz.isInstance(rawValue)) {
      return clazz.cast(rawValue);
    }
    if (defaultValue != null
        && (rawValue instanceof StarlarkList<?> listValue)
        && listValue.isEmpty()) {
      return clazz.cast(defaultValue);
    }
    throw new ProjectFunctionException(
        new TypecheckFailureException(
            String.format("%s, got %s", errorMessage, rawValue.getClass())));
  }
}
