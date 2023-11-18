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
package com.google.devtools.build.lib.skyframe.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertNoEvents;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Simple tests for {@link BazelPackageLoader}.
 *
 * <p>Bazel's unit and integration tests do consistency checks with {@link BazelPackageLoader} under
 * the covers, so we get pretty exhaustive correctness tests for free.
 */
@RunWith(JUnit4.class)
public final class BazelPackageLoaderTest extends AbstractPackageLoaderTest {

  private Path installBase;
  private Path outputBase;
  private Path rulesJavaWorkspace;

  @Before
  public void setUp() throws Exception {
    installBase = fs.getPath("/installBase/");
    installBase.createDirectoryAndParents();
    outputBase = fs.getPath("/outputBase/");
    outputBase.createDirectoryAndParents();
    Path embeddedBinaries = ServerDirectories.getEmbeddedBinariesRoot(installBase);
    embeddedBinaries.createDirectoryAndParents();

    mockEmbeddedTools(embeddedBinaries);
    fetchExternalRepo(RepositoryName.create("bazel_tools"));

    createWorkspaceFile("");
  }

  private String getDefaultWorkspaceContent() {
    // Skip the WORKSPACE suffix to avoid loading rules_java
    return "# __SKIP_WORKSPACE_SUFFIX__";
  }

  private void createWorkspaceFile(String content) throws Exception {
    file("WORKSPACE", getDefaultWorkspaceContent(), content);
  }

  private static void mockEmbeddedTools(Path embeddedBinaries) throws IOException {
    Path tools = embeddedBinaries.getRelative("embedded_tools");
    tools.getRelative("tools/cpp").createDirectoryAndParents();
    tools.getRelative("tools/osx").createDirectoryAndParents();
    FileSystemUtils.writeIsoLatin1(tools.getRelative("WORKSPACE"), "");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/cpp/BUILD"), "");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/cpp/cc_configure.bzl"),
        "def cc_configure(*args, **kwargs):",
        "    pass");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/osx/BUILD"), "");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/osx/xcode_configure.bzl"),
        "def xcode_configure(*args, **kwargs):",
        "    pass");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/sh/BUILD"), "");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/sh/sh_configure.bzl"),
        "def sh_configure(*args, **kwargs):",
        "    pass");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/build_defs/repo/BUILD"));
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/build_defs/repo/http.bzl"),
        "def http_archive(**kwargs):",
        "  pass",
        "",
        "def http_file(**kwargs):",
        "  pass",
        "",
        "def http_jar(**kwargs):",
        "  pass");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/build_defs/repo/utils.bzl"),
        "def maybe(repo_rule, name, **kwargs):",
        "  if name not in native.existing_rules():",
        "    repo_rule(name = name, **kwargs)");
    FileSystemUtils.writeIsoLatin1(tools.getRelative("tools/jdk/BUILD"));
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/jdk/jdk_build_file.bzl"), "JDK_BUILD_TEMPLATE = ''");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/jdk/local_java_repository.bzl"),
        "def local_java_repository(**kwargs):",
        "  pass");
    FileSystemUtils.writeIsoLatin1(
        tools.getRelative("tools/jdk/remote_java_repository.bzl"),
        "def remote_java_repository(**kwargs):",
        "  pass");
  }

  private void fetchExternalRepo(RepositoryName externalRepo) {
    try (PackageLoader pkgLoaderForFetch =
        newPackageLoaderBuilder(root).setFetchForTesting().useDefaultStarlarkSemantics().build()) {
      // Load the package '' in this repo. This package may or may not exist; we don't care since we
      // merely need the side-effects of the 'fetch' work.
      PackageIdentifier pkgId = PackageIdentifier.create(externalRepo, PathFragment.create(""));
      try {
        pkgLoaderForFetch.loadPackage(pkgId);
      } catch (NoSuchPackageException | InterruptedException e) {
        // Doesn't matter; see above comment.
      }
    }
  }

  @Override
  protected BazelPackageLoader.Builder newPackageLoaderBuilder(Root workspaceDir) {
    return BazelPackageLoader.builder(workspaceDir, installBase, outputBase);
  }

  @Override
  protected ForkJoinPool extractLegacyGlobbingForkJoinPool(PackageLoader packageLoader) {
    return ((BazelPackageLoader) packageLoader).forkJoinPoolForNonSkyframeGlobbing;
  }

  @Test
  public void simpleLocalRepositoryPackage() throws Exception {
    createWorkspaceFile("local_repository(name = 'r', path='r')");
    file("r/WORKSPACE", "workspace(name = 'r')");
    file("r/good/BUILD", "sh_library(name = 'good')");
    RepositoryName rRepoName = RepositoryName.create("r");
    fetchExternalRepo(rRepoName);

    PackageIdentifier pkgId = PackageIdentifier.create(rRepoName, PathFragment.create("good"));
    Package goodPkg;
    try (PackageLoader pkgLoader = newPackageLoader()) {
      goodPkg = pkgLoader.loadPackage(pkgId);
    }
    assertThat(goodPkg.containsErrors()).isFalse();
    assertThat(goodPkg.getTarget("good").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void newLocalRepository() throws Exception {
    createWorkspaceFile(
        "new_local_repository(name = 'r', path = '/r', "
            + "build_file_content = 'sh_library(name = \"good\")')");
    fs.getPath("/r").createDirectoryAndParents();
    RepositoryName rRepoName = RepositoryName.create("r");
    fetchExternalRepo(rRepoName);

    PackageIdentifier pkgId =
        PackageIdentifier.create(rRepoName, PathFragment.create(""));
    Package goodPkg;
    try (PackageLoader pkgLoader = newPackageLoader()) {
      goodPkg = pkgLoader.loadPackage(pkgId);
    }
    assertThat(goodPkg.containsErrors()).isFalse();
    assertThat(goodPkg.getTarget("good").getAssociatedRule().getRuleClass())
        .isEqualTo("sh_library");
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void buildDotBazelForSubpackageCheckDuringGlobbing() throws Exception {
    file("a/BUILD", "filegroup(name = 'fg', srcs = glob(['sub/a.txt'], allow_empty = True))");
    file("a/sub/a.txt");
    file("a/sub/BUILD.bazel");

    PackageIdentifier pkgId = PackageIdentifier.createInMainRepo(PathFragment.create("a"));
    Package aPkg;
    try (PackageLoader pkgLoader = newPackageLoader()) {
      aPkg = pkgLoader.loadPackage(pkgId);
    }
    assertThat(aPkg.containsErrors()).isFalse();
    assertThrows(NoSuchTargetException.class, () -> aPkg.getTarget("sub/a.txt"));
    assertNoEvents(handler.getEvents());
  }

  @Test
  public void incompatibleOptionsPreservedInExec() throws IllegalAccessException {
    ImmutableMultimap.Builder<Class<? extends FragmentOptions>, OptionDefinition>
        missingMetadataTagOptions = new ImmutableMultimap.Builder<>();
    ImmutableMultimap.Builder<Class<? extends FragmentOptions>, OptionDefinition>
        unpreservedOptions = new ImmutableMultimap.Builder<>();
    ImmutableSortedSet<Class<? extends FragmentOptions>> allFragmentOptions =
        newPackageLoaderBuilder().ruleClassProvider.getFragmentRegistry().getOptionsClasses();
    for (Class<? extends FragmentOptions> optionsClass : allFragmentOptions) {
      ImmutableList<OptionDefinition> incompatibleOptions =
          OptionsParser.getOptionDefinitions(optionsClass).stream()
              .filter(
                  option ->
                      Arrays.asList(option.getOptionMetadataTags())
                              .contains(OptionMetadataTag.INCOMPATIBLE_CHANGE)
                          || option.getOptionName().startsWith("incompatible_"))
              .filter(option -> option.getField().getType().isAssignableFrom(boolean.class))
              .filter(option -> option.getField().getAnnotation(Deprecated.class) == null)
              .collect(ImmutableList.toImmutableList());

      // Verify that all --incompatible_* options have the INCOMPATIBLE_CHANGE metadata tag.
      incompatibleOptions.stream()
          .filter(
              option ->
                  !Arrays.asList(option.getOptionMetadataTags())
                      .contains(OptionMetadataTag.INCOMPATIBLE_CHANGE))
          .forEach(option -> missingMetadataTagOptions.put(optionsClass, option));

      // Flip all incompatible (boolean) options to their non-default value.
      FragmentOptions flipped = Options.getDefaults(optionsClass);
      for (OptionDefinition incompatibleOption : incompatibleOptions) {
        Field field = incompatibleOption.getField();
        field.setBoolean(flipped, !field.getBoolean(flipped));
      }

      // Verify that the flipped value is preserved under an exec transition.
      FragmentOptions flippedAfterExec = flipped.getExec();
      for (OptionDefinition incompatibleOption : incompatibleOptions) {
        Field field = incompatibleOption.getField();
        if (field.getBoolean(flippedAfterExec) != field.getBoolean(flipped)) {
          unpreservedOptions.put(optionsClass, incompatibleOption);
        }
      }
    }

    assertThat(missingMetadataTagOptions.build()).isEmpty();
    assertThat(unpreservedOptions.build()).isEmpty();
  }
}
