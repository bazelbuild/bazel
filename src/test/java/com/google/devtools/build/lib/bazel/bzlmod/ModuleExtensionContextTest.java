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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.buildModule;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createTagClass;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.util.Optional;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ModuleExtensionContext}. */
@RunWith(JUnit4.class)
public class ModuleExtensionContextTest {

  private ModuleExtensionContext createTestContext(
      ModuleExtensionId extensionId, StarlarkList<StarlarkBazelModule> modules) throws Exception {
    InMemoryFileSystem fileSystem = new InMemoryFileSystem();
    Path workingDirectory = fileSystem.getPath("/working");
    workingDirectory.createDirectoryAndParents();
    
    BlazeDirectories directories =
        new BlazeDirectories(
            /* serverDirectories= */ null,
            workingDirectory,
            /* defaultSystemJavabase= */ null,
            /* productName= */ "bazel");

    Environment env = SkyframeExecutorTestUtils.getDummyEnv();
    
    return new ModuleExtensionContext(
        workingDirectory,
        directories,
        env,
        /* envVariables= */ ImmutableMap.of(),
        /* downloadManager= */ new DownloadManager(null, null),
        /* timeoutScaling= */ 1.0,
        /* processWrapper= */ null,
        StarlarkSemantics.DEFAULT,
        /* remoteExecutor= */ null,
        extensionId,
        modules,
        /* rootModuleHasNonDevDependency= */ true);
  }

  @Test
  public void testGetRootModule() throws Exception {
    // Create test modules: root module and a regular module
    ModuleKey rootKey = createModuleKey("root_module", "1.0");
    ModuleKey regularKey = createModuleKey("regular_module", "2.0");
    
    Module rootModule = buildModule("root_module", "1.0").setKey(rootKey).build();
    Module regularModule = buildModule("regular_module", "2.0").setKey(regularKey).build();
    
    AbridgedModule rootAbridged = AbridgedModule.from(rootModule);
    AbridgedModule regularAbridged = AbridgedModule.from(regularModule);
    
    // Create test extension
    ModuleExtension extension =
        ModuleExtension.builder()
            .setDoc(Optional.empty())
            .setDefiningBzlFileLabel(Label.parseCanonicalUnchecked("@root_module//:extension.bzl"))
            .setLocation(Location.BUILTIN)
            .setImplementation(() -> "test_extension")
            .setEnvVariables(ImmutableList.of())
            .setOsDependent(false)
            .setArchDependent(false)
            .setTagClasses(ImmutableMap.of("test", createTagClass()))
            .build();
    
    // Create Starlark modules - root module is marked as root
    StarlarkBazelModule rootStarlarkModule =
        StarlarkBazelModule.create(
            rootAbridged,
            extension,
            rootModule.getRepoMappingWithBazelDepsOnly(ImmutableMap.of()),
            null,
            new Label.RepoMappingRecorder());
            
    StarlarkBazelModule regularStarlarkModule =
        StarlarkBazelModule.create(
            regularAbridged,
            extension,
            regularModule.getRepoMappingWithBazelDepsOnly(ImmutableMap.of()),
            null,
            new Label.RepoMappingRecorder());

    ModuleExtensionId extensionId =
        ModuleExtensionId.create(
            Label.parseCanonicalUnchecked("@root_module//:extension.bzl"),
            "test_extension",
            Optional.empty());

    StarlarkList<StarlarkBazelModule> modules =
        StarlarkList.immutableOf(rootStarlarkModule, regularStarlarkModule);

    ModuleExtensionContext context = createTestContext(extensionId, modules);

    // Test root_module property
    StarlarkBazelModule result = context.getRootModule();
    assertThat(result).isNotNull();
    assertThat(result.getName()).isEqualTo("root_module");
    assertThat(result.isRoot()).isTrue();
  }

  @Test
  public void testGetRootModuleWhenNoRootExists() throws Exception {
    // Create only regular modules, no root module
    ModuleKey regularKey = createModuleKey("regular_module", "2.0");
    Module regularModule = buildModule("regular_module", "2.0").setKey(regularKey).build();
    AbridgedModule regularAbridged = AbridgedModule.from(regularModule);
    
    ModuleExtension extension =
        ModuleExtension.builder()
            .setDoc(Optional.empty())
            .setDefiningBzlFileLabel(Label.parseCanonicalUnchecked("@regular_module//:extension.bzl"))
            .setLocation(Location.BUILTIN)
            .setImplementation(() -> "test_extension")
            .setEnvVariables(ImmutableList.of())
            .setOsDependent(false)
            .setArchDependent(false)
            .setTagClasses(ImmutableMap.of("test", createTagClass()))
            .build();
    
    StarlarkBazelModule regularStarlarkModule =
        StarlarkBazelModule.create(
            regularAbridged,
            extension,
            regularModule.getRepoMappingWithBazelDepsOnly(ImmutableMap.of()),
            null,
            new Label.RepoMappingRecorder());

    ModuleExtensionId extensionId =
        ModuleExtensionId.create(
            Label.parseCanonicalUnchecked("@regular_module//:extension.bzl"),
            "test_extension",
            Optional.empty());

    StarlarkList<StarlarkBazelModule> modules = StarlarkList.immutableOf(regularStarlarkModule);
    ModuleExtensionContext context = createTestContext(extensionId, modules);

    // Test root_module property returns null when no root module exists
    StarlarkBazelModule result = context.getRootModule();
    assertThat(result).isNull();
  }

  @Test
  public void testGetCurrentModule() throws Exception {
    // Create modules
    ModuleKey ownerKey = createModuleKey("owner_module", "1.0");
    ModuleKey otherKey = createModuleKey("other_module", "2.0");
    
    Module ownerModule = buildModule("owner_module", "1.0").setKey(ownerKey).build();
    Module otherModule = buildModule("other_module", "2.0").setKey(otherKey).build();
    
    AbridgedModule ownerAbridged = AbridgedModule.from(ownerModule);
    AbridgedModule otherAbridged = AbridgedModule.from(otherModule);
    
    ModuleExtension extension =
        ModuleExtension.builder()
            .setDoc(Optional.empty())
            .setDefiningBzlFileLabel(Label.parseCanonicalUnchecked("@owner_module//:extension.bzl"))
            .setLocation(Location.BUILTIN)
            .setImplementation(() -> "test_extension")
            .setEnvVariables(ImmutableList.of())
            .setOsDependent(false)
            .setArchDependent(false)
            .setTagClasses(ImmutableMap.of("test", createTagClass()))
            .build();
    
    StarlarkBazelModule ownerStarlarkModule =
        StarlarkBazelModule.create(
            ownerAbridged,
            extension,
            ownerModule.getRepoMappingWithBazelDepsOnly(ImmutableMap.of()),
            null,
            new Label.RepoMappingRecorder());
            
    StarlarkBazelModule otherStarlarkModule =
        StarlarkBazelModule.create(
            otherAbridged,
            extension,
            otherModule.getRepoMappingWithBazelDepsOnly(ImmutableMap.of()),
            null,
            new Label.RepoMappingRecorder());

    // Extension is defined in owner_module
    ModuleExtensionId extensionId =
        ModuleExtensionId.create(
            Label.parseCanonicalUnchecked("@owner_module//:extension.bzl"),
            "test_extension",
            Optional.empty());

    StarlarkList<StarlarkBazelModule> modules =
        StarlarkList.immutableOf(ownerStarlarkModule, otherStarlarkModule);

    ModuleExtensionContext context = createTestContext(extensionId, modules);

    // Test current_module property returns the module that defined the extension
    StarlarkBazelModule result = context.getCurrentModule();
    assertThat(result).isNotNull();
    assertThat(result.getName()).isEqualTo("owner_module");
  }

  @Test
  public void testGetCurrentModuleWhenOwnerNotInList() throws Exception {
    // Create a module that is different from the extension owner
    ModuleKey otherKey = createModuleKey("other_module", "2.0");
    Module otherModule = buildModule("other_module", "2.0").setKey(otherKey).build();
    AbridgedModule otherAbridged = AbridgedModule.from(otherModule);
    
    ModuleExtension extension =
        ModuleExtension.builder()
            .setDoc(Optional.empty())
            .setDefiningBzlFileLabel(Label.parseCanonicalUnchecked("@owner_module//:extension.bzl"))
            .setLocation(Location.BUILTIN)
            .setImplementation(() -> "test_extension")
            .setEnvVariables(ImmutableList.of())
            .setOsDependent(false)
            .setArchDependent(false)
            .setTagClasses(ImmutableMap.of("test", createTagClass()))
            .build();
    
    StarlarkBazelModule otherStarlarkModule =
        StarlarkBazelModule.create(
            otherAbridged,
            extension,
            otherModule.getRepoMappingWithBazelDepsOnly(ImmutableMap.of()),
            null,
            new Label.RepoMappingRecorder());

    // Extension is defined in owner_module, but only other_module is in the list
    ModuleExtensionId extensionId =
        ModuleExtensionId.create(
            Label.parseCanonicalUnchecked("@owner_module//:extension.bzl"),
            "test_extension",
            Optional.empty());

    StarlarkList<StarlarkBazelModule> modules = StarlarkList.immutableOf(otherStarlarkModule);
    ModuleExtensionContext context = createTestContext(extensionId, modules);

    // Test current_module property returns null when owner module is not in the list
    StarlarkBazelModule result = context.getCurrentModule();
    assertThat(result).isNull();
  }
}