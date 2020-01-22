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

import static com.google.devtools.build.lib.syntax.Starlark.NONE;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.skylarkbuildapi.WorkspaceGlobalsApi;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** A collection of global skylark build API functions that apply to WORKSPACE files. */
public class WorkspaceGlobals implements WorkspaceGlobalsApi {

  // Must start with a letter and can contain letters, numbers, and underscores
  private static final Pattern LEGAL_WORKSPACE_NAME = Pattern.compile("^\\p{Alpha}\\w*$");

  private final boolean allowOverride;
  private final RuleFactory ruleFactory;
  // Mapping of the relative paths of the incrementally updated managed directories
  // to the managing external repositories
  private final TreeMap<PathFragment, RepositoryName> managedDirectoriesMap;
  // Directories to be excluded from symlinking to the execroot.
  private ImmutableSortedSet<String> doNotSymlinkInExecrootPaths;

  public WorkspaceGlobals(boolean allowOverride, RuleFactory ruleFactory) {
    this.allowOverride = allowOverride;
    this.ruleFactory = ruleFactory;
    this.managedDirectoriesMap = Maps.newTreeMap();
  }

  @Override
  public NoneType workspace(
      String name,
      Dict<?, ?> managedDirectories, // <String, Object>
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    if (allowOverride) {
      if (!isLegalWorkspaceName(name)) {
        throw Starlark.errorf("%s is not a legal workspace name", name);
      }
      String errorMessage = LabelValidator.validateTargetName(name);
      if (errorMessage != null) {
        throw Starlark.errorf("%s", errorMessage);
      }
      PackageFactory.getContext(thread).pkgBuilder.setWorkspaceName(name);
      Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
      RuleClass localRepositoryRuleClass = ruleFactory.getRuleClass("local_repository");
      RuleClass bindRuleClass = ruleFactory.getRuleClass("bind");
      Map<String, Object> kwargs = ImmutableMap.<String, Object>of("name", name, "path", ".");
      try {
        // This effectively adds a "local_repository(name = "<ws>", path = ".")"
        // definition to the WORKSPACE file.
        WorkspaceFactoryHelper.createAndAddRepositoryRule(
            builder, localRepositoryRuleClass, bindRuleClass, kwargs, thread.getCallerLocation());
      } catch (InvalidRuleException | NameConflictException | LabelSyntaxException e) {
        throw Starlark.errorf("%s", e.getMessage());
      }
      // Add entry in repository map from "@name" --> "@" to avoid issue where bazel
      // treats references to @name as a separate external repo
      builder.addRepositoryMappingEntry(
          RepositoryName.MAIN,
          RepositoryName.createFromValidStrippedName(name),
          RepositoryName.MAIN);
      parseManagedDirectories(
          managedDirectories.getContents(String.class, Object.class, "managed_directories"));
      return NONE;
    } else {
      throw Starlark.errorf(
          "workspace() function should be used only at the top of the WORKSPACE file");
    }
  }

  @Override
  public NoneType dontSymlinkDirectoriesInExecroot(Sequence<?> paths, StarlarkThread thread)
      throws EvalException, InterruptedException {
    List<String> pathsList = paths.getContents(String.class, "paths");
    Set<String> set = Sets.newHashSet();
    for (String path : pathsList) {
      PathFragment pathFragment = PathFragment.create(path);
      if (pathFragment.isEmpty()) {
        throw Starlark.errorf(
            "Empty path can not be passed to dont_symlink_directories_in_execroot.");
      }
      if (pathFragment.containsUplevelReferences() || pathFragment.segmentCount() > 1) {
        throw Starlark.errorf(
            "dont_symlink_directories_in_execroot can only accept top level directories under"
                + " workspace, \"%s\" can not be specified as an attribute.",
            path);
      }
      if (pathFragment.isAbsolute()) {
        throw Starlark.errorf(
            "dont_symlink_directories_in_execroot can only accept top level directories under"
                + " workspace, absolute path \"%s\" can not be specified as an attribute.",
            path);
      }
      if (!set.add(pathFragment.getBaseName())) {
        throw Starlark.errorf(
            "dont_symlink_directories_in_execroot should not contain duplicate values: \"%s\" is"
                + " specified more then once.",
            path);
      }
    }
    doNotSymlinkInExecrootPaths = ImmutableSortedSet.copyOf(set);
    return NONE;
  }

  private void parseManagedDirectories(
      Map<String, ?> managedDirectories) // <String, Sequence<String>>
      throws EvalException {
    Map<PathFragment, String> nonNormalizedPathsMap = Maps.newHashMap();
    for (Map.Entry<String, ?> entry : managedDirectories.entrySet()) {
      RepositoryName repositoryName = createRepositoryName(entry.getKey());
      List<PathFragment> paths =
          getManagedDirectoriesPaths(entry.getValue(), nonNormalizedPathsMap);
      for (PathFragment dir : paths) {
        PathFragment floorKey = managedDirectoriesMap.floorKey(dir);
        if (dir.equals(floorKey)) {
          throw Starlark.errorf(
              "managed_directories attribute should not contain multiple"
                  + " (or duplicate) repository mappings for the same directory ('%s').",
              nonNormalizedPathsMap.get(dir));
        }
        PathFragment ceilingKey = managedDirectoriesMap.ceilingKey(dir);
        boolean isDescendant = floorKey != null && dir.startsWith(floorKey);
        if (isDescendant || (ceilingKey != null && ceilingKey.startsWith(dir))) {
          throw Starlark.errorf(
              "managed_directories attribute value can not contain nested mappings."
                  + " '%s' is a descendant of '%s'.",
              nonNormalizedPathsMap.get(isDescendant ? dir : ceilingKey),
              nonNormalizedPathsMap.get(isDescendant ? floorKey : dir));
        }
        managedDirectoriesMap.put(dir, repositoryName);
      }
    }
  }

  private static RepositoryName createRepositoryName(String key) throws EvalException {
    if (!key.startsWith("@")) {
      throw Starlark.errorf(
          "Cannot parse repository name '%s'. Repository name should start with '@'.", key);
    }
    try {
      return RepositoryName.create(key);
    } catch (LabelSyntaxException e) {
      throw Starlark.errorf("%s", e);
    }
  }

  private static List<PathFragment> getManagedDirectoriesPaths(
      Object directoriesList, Map<PathFragment, String> nonNormalizedPathsMap)
      throws EvalException {
    if (!(directoriesList instanceof Sequence)) {
      throw Starlark.errorf(
          "managed_directories attribute value should be of the type attr.string_list_dict(),"
              + " mapping repository name to the list of managed directories.");
    }
    List<PathFragment> result = Lists.newArrayList();
    for (Object obj : (Sequence) directoriesList) {
      if (!(obj instanceof String)) {
        throw Starlark.errorf("Expected managed directory path (as string), but got '%s'.", obj);
      }
      String path = ((String) obj).trim();
      if (path.isEmpty()) {
        throw Starlark.errorf("Expected managed directory path to be non-empty string.");
      }
      PathFragment pathFragment = PathFragment.create(path);
      if (pathFragment.isAbsolute()) {
        throw Starlark.errorf(
            "Expected managed directory path ('%s') to be relative to the workspace root.", path);
      }
      if (pathFragment.containsUplevelReferences()) {
        throw Starlark.errorf(
            "Expected managed directory path ('%s') to be under the workspace root.", path);
      }
      nonNormalizedPathsMap.put(pathFragment, path);
      result.add(pathFragment);
    }
    return result;
  }

  public Map<PathFragment, RepositoryName> getManagedDirectories() {
    return managedDirectoriesMap;
  }

  public ImmutableSortedSet<String> getDoNotSymlinkInExecrootPaths() {
    return doNotSymlinkInExecrootPaths;
  }

  private static RepositoryName getRepositoryName(@Nullable Label label) {
    if (label == null) {
      // registration happened directly in the main WORKSPACE
      return RepositoryName.MAIN;
    }

    // registeration happened in a loaded bzl file
    return label.getPackageIdentifier().getRepository();
  }

  private static List<String> renamePatterns(
      List<String> patterns, Package.Builder builder, StarlarkThread thread) {
    RepositoryName myName = getRepositoryName((Label) thread.getGlobals().getLabel());
    Map<RepositoryName, RepositoryName> renaming = builder.getRepositoryMappingFor(myName);
    return patterns.stream()
        .map(patternEntry -> TargetPattern.renameRepository(patternEntry, renaming))
        .collect(ImmutableList.toImmutableList());
  }

  @Override
  public NoneType registerExecutionPlatforms(Sequence<?> platformLabels, StarlarkThread thread)
      throws EvalException, InterruptedException {
    // Add to the package definition for later.
    Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
    List<String> patterns = platformLabels.getContents(String.class, "platform_labels");
    builder.addRegisteredExecutionPlatforms(renamePatterns(patterns, builder, thread));
    return NONE;
  }

  @Override
  public NoneType registerToolchains(Sequence<?> toolchainLabels, StarlarkThread thread)
      throws EvalException, InterruptedException {
    // Add to the package definition for later.
    Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
    List<String> patterns = toolchainLabels.getContents(String.class, "toolchain_labels");
    builder.addRegisteredToolchains(renamePatterns(patterns, builder, thread));
    return NONE;
  }

  @Override
  public NoneType bind(String name, Object actual, StarlarkThread thread)
      throws EvalException, InterruptedException {
    Label nameLabel;
    try {
      nameLabel = Label.parseAbsolute("//external:" + name, ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
    try {
      Package.Builder builder = PackageFactory.getContext(thread).pkgBuilder;
      RuleClass ruleClass = ruleFactory.getRuleClass("bind");
      WorkspaceFactoryHelper.addBindRule(
          builder,
          ruleClass,
          nameLabel,
          actual == NONE ? null : Label.parseAbsolute((String) actual, ImmutableMap.of()),
          thread.getCallerLocation(),
          new AttributeContainer(ruleClass));
    } catch (RuleFactory.InvalidRuleException
        | Package.NameConflictException
        | LabelSyntaxException e) {
      throw Starlark.errorf("%s", e.getMessage());
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
