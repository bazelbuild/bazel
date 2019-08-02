// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import static com.google.devtools.build.lib.syntax.Runtime.NONE;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.skylarkbuildapi.WorkspaceGlobalsApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** A collection of global skylark build API functions that apply to WORKSPACE files. */
public class WorkspaceGlobals implements WorkspaceGlobalsApi {

  // Must start with a letter and can contain letters, numbers, and underscores
  private static final Pattern LEGAL_WORKSPACE_NAME = Pattern.compile("^\\p{Alpha}\\w*$");

  private final boolean allowOverride;
  private final RuleFactory ruleFactory;
  // Mapping of the relative paths of the incrementally updated managed directories
  // to the managing external repositories
  private final TreeMap<PathFragment, RepositoryName> managedDirectoriesMap;

  public WorkspaceGlobals(boolean allowOverride, RuleFactory ruleFactory) {
    this.allowOverride = allowOverride;
    this.ruleFactory = ruleFactory;
    this.managedDirectoriesMap = Maps.newTreeMap();
  }

  @Override
  public NoneType workspace(
      String name,
      SkylarkDict<String, Object> managedDirectories,
      FuncallExpression ast,
      Environment env)
      throws EvalException, InterruptedException {
    if (allowOverride) {
      if (!isLegalWorkspaceName(name)) {
        throw new EvalException(ast.getLocation(), name + " is not a legal workspace name");
      }
      String errorMessage = LabelValidator.validateTargetName(name);
      if (errorMessage != null) {
        throw new EvalException(ast.getLocation(), errorMessage);
      }
      PackageFactory.getContext(env, ast.getLocation()).pkgBuilder.setWorkspaceName(name);
      Package.Builder builder = PackageFactory.getContext(env, ast.getLocation()).pkgBuilder;
      RuleClass localRepositoryRuleClass = ruleFactory.getRuleClass("local_repository");
      RuleClass bindRuleClass = ruleFactory.getRuleClass("bind");
      Map<String, Object> kwargs = ImmutableMap.<String, Object>of("name", name, "path", ".");
      try {
        // This effectively adds a "local_repository(name = "<ws>", path = ".")"
        // definition to the WORKSPACE file.
        WorkspaceFactoryHelper.createAndAddRepositoryRule(
            builder, localRepositoryRuleClass, bindRuleClass, kwargs, ast);
      } catch (InvalidRuleException | NameConflictException | LabelSyntaxException e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
      // Add entry in repository map from "@name" --> "@" to avoid issue where bazel
      // treats references to @name as a separate external repo
      if (env.getSemantics().incompatibleRemapMainRepo()) {
        builder.addRepositoryMappingEntry(
            RepositoryName.MAIN,
            RepositoryName.createFromValidStrippedName(name),
            RepositoryName.MAIN);
      }
      parseManagedDirectories(managedDirectories, ast);
      return NONE;
    } else {
      throw new EvalException(
          ast.getLocation(),
          "workspace() function should be used only at the top of the WORKSPACE file");
    }
  }

  private void parseManagedDirectories(
      SkylarkDict<String, Object> managedDirectories, FuncallExpression ast) throws EvalException {
    Map<PathFragment, String> nonNormalizedPathsMap = Maps.newHashMap();
    for (Map.Entry<String, Object> entry : managedDirectories.entrySet()) {
      RepositoryName repositoryName = createRepositoryName(entry.getKey(), ast.getLocation());
      List<PathFragment> paths =
          getManagedDirectoriesPaths(entry.getValue(), ast.getLocation(), nonNormalizedPathsMap);
      for (PathFragment dir : paths) {
        PathFragment floorKey = managedDirectoriesMap.floorKey(dir);
        if (dir.equals(floorKey)) {
          throw new EvalException(
              ast.getLocation(),
              String.format(
                  "managed_directories attribute should not contain multiple"
                      + " (or duplicate) repository mappings for the same directory ('%s').",
                  nonNormalizedPathsMap.get(dir)));
        }
        PathFragment ceilingKey = managedDirectoriesMap.ceilingKey(dir);
        boolean isDescendant = floorKey != null && dir.startsWith(floorKey);
        if (isDescendant || (ceilingKey != null && ceilingKey.startsWith(dir))) {
          throw new EvalException(
              ast.getLocation(),
              String.format(
                  "managed_directories attribute value can not contain nested mappings."
                      + " '%s' is a descendant of '%s'.",
                  nonNormalizedPathsMap.get(isDescendant ? dir : ceilingKey),
                  nonNormalizedPathsMap.get(isDescendant ? floorKey : dir)));
        }
        managedDirectoriesMap.put(dir, repositoryName);
      }
    }
  }

  private RepositoryName createRepositoryName(String key, Location location) throws EvalException {
    if (!key.startsWith("@")) {
      throw new EvalException(
          location,
          String.format(
              "Cannot parse repository name '%s'. Repository name should start with '@'.", key));
    }
    try {
      return RepositoryName.create(key);
    } catch (LabelSyntaxException e) {
      throw new EvalException(location, e);
    }
  }

  private List<PathFragment> getManagedDirectoriesPaths(
      Object directoriesList, Location location, Map<PathFragment, String> nonNormalizedPathsMap)
      throws EvalException {
    if (!(directoriesList instanceof SkylarkList)) {
      throw new EvalException(
          location,
          "managed_directories attribute value should be of the type attr.string_list_dict(),"
              + " mapping repository name to the list of managed directories.");
    }
    List<PathFragment> result = Lists.newArrayList();
    for (Object obj : (SkylarkList) directoriesList) {
      if (!(obj instanceof String)) {
        throw new EvalException(
            location,
            String.format("Expected managed directory path (as string), but got '%s'.", obj));
      }
      String path = ((String) obj).trim();
      if (path.isEmpty()) {
        throw new EvalException(
            location, "Expected managed directory path to be non-empty string.");
      }
      PathFragment pathFragment = PathFragment.create(path);
      if (pathFragment.isAbsolute()) {
        throw new EvalException(
            location,
            String.format(
                "Expected managed directory path ('%s') to be relative to the workspace root.",
                path));
      }
      if (pathFragment.containsUplevelReferences()) {
        throw new EvalException(
            location,
            String.format(
                "Expected managed directory path ('%s') to be under the workspace root.", path));
      }
      nonNormalizedPathsMap.put(pathFragment, path);
      result.add(pathFragment);
    }
    return result;
  }

  public Map<PathFragment, RepositoryName> getManagedDirectories() {
    return managedDirectoriesMap;
  }

  @Override
  public NoneType registerExecutionPlatforms(
      SkylarkList<?> platformLabels, Location location, Environment env)
      throws EvalException, InterruptedException {
    // Add to the package definition for later.
    Package.Builder builder = PackageFactory.getContext(env, location).pkgBuilder;
    builder.addRegisteredExecutionPlatforms(
        platformLabels.getContents(String.class, "platform_labels"));

    return NONE;
  }

  @Override
  public NoneType registerToolchains(
      SkylarkList<?> toolchainLabels, Location location, Environment env)
      throws EvalException, InterruptedException {
    // Add to the package definition for later.
    Package.Builder builder = PackageFactory.getContext(env, location).pkgBuilder;
    builder.addRegisteredToolchains(toolchainLabels.getContents(String.class, "toolchain_labels"));

    return NONE;
  }

  @Override
  public NoneType bind(String name, Object actual, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    Label nameLabel;
    try {
      nameLabel = Label.parseAbsolute("//external:" + name, ImmutableMap.of());
      try {
        Package.Builder builder = PackageFactory.getContext(env, ast.getLocation()).pkgBuilder;
        RuleClass ruleClass = ruleFactory.getRuleClass("bind");
        WorkspaceFactoryHelper.addBindRule(
            builder,
            ruleClass,
            nameLabel,
            actual == NONE ? null : Label.parseAbsolute((String) actual, ImmutableMap.of()),
            ast.getLocation(),
            ruleFactory.getAttributeContainer(ruleClass));
      } catch (RuleFactory.InvalidRuleException
          | Package.NameConflictException
          | LabelSyntaxException e) {
        throw new EvalException(ast.getLocation(), e.getMessage());
      }

    } catch (LabelSyntaxException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
    return NONE;
  }

  /**
   * Returns true if the given name is a valid workspace name.
   */
  public static boolean isLegalWorkspaceName(String name) {
    Matcher matcher = LEGAL_WORKSPACE_NAME.matcher(name);
    return matcher.matches();
  }
}
