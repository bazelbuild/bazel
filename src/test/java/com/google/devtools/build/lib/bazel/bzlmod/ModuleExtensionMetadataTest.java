package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.EventKind;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ModuleExtensionMetadataTest {
  @Test
  public void testCreate_exactlyOneArgIsNone() {
    Exception e = assertThrows(
        EvalException.class,
        () -> ModuleExtensionMetadata.create(Starlark.NONE, StarlarkList.immutableOf()));
    assertThat(e).hasMessageThat().isEqualTo(
        "root_module_direct_deps and root_module_direct_dev_deps must both be specified or both be unspecified");

    e = assertThrows(
        EvalException.class,
        () -> ModuleExtensionMetadata.create(StarlarkList.immutableOf(), Starlark.NONE));
    assertThat(e).hasMessageThat().isEqualTo(
        "root_module_direct_deps and root_module_direct_dev_deps must both be specified or both be unspecified");
  }

  private static final Location TEST_ROOT_LOCATION = Location.fromFileLineColumn("<root>/MODULE.bazel", 2, 1);
  private static final ImmutableList<ModuleExtensionUsage> TEST_USAGES = ImmutableList.of(
      ModuleExtensionUsage.builder()
          .setExtensionBzlFile("@mod//:extensions.bzl")
          .setExtensionName("ext")
          .setLocation(TEST_ROOT_LOCATION)
          .setImports(ImmutableBiMap.of("indirect_dep", "indirect_dep", "indirect_dev_dep",
              "indirect_dev_dep", "invalid_dep", "invalid_dep", "invalid_dev_dep", "invalid_dev_dep"))
          .setDevImports(ImmutableSet.of("indirect_dev_dep", "invalid_dev_dep"))
          .build(),
      ModuleExtensionUsage.builder()
          .setExtensionBzlFile("@mod//:extensions.bzl")
          .setExtensionName("ext")
          .setLocation(Location.fromFileLineColumn("mod@1.2.3/MODULE.bazel", 2, 1))
          .setImports(ImmutableBiMap.of("indirect_dep", "indirect_dep"))
          .setDevImports(ImmutableSet.of())
          .build()
  );
  private static final ImmutableSet<String> TEST_ALL_REPOS = ImmutableSet.of("direct_dep",
      "direct_dev_dep", "indirect_dep", "indirect_dev_dep");

  @Test
  public void testGenerateFixupMessage() throws EvalException {
    var moduleExtensionMetadata = ModuleExtensionMetadata.create(
        StarlarkList.immutableOf("direct_dep"),
        StarlarkList.immutableOf("direct_dev_dep"));
    var fixupMessage = moduleExtensionMetadata.generateFixupMessage(TEST_USAGES, TEST_ALL_REPOS);

    assertThat(fixupMessage.isPresent()).isTrue();
    assertThat(fixupMessage.get().getLocation()).isEqualTo(TEST_ROOT_LOCATION);
    assertThat(fixupMessage.get().getKind()).isEqualTo(EventKind.WARNING);
    assertThat(fixupMessage.get().getMessage()).isEqualTo(
        "The module extension ext defined in @mod//:extensions.bzl reported incorrect imports of repositories via use_repo():\n"
            + "\n"
            + "Imported, but not created by the extension (will cause the build to fail):\n"
            + "    invalid_dep, invalid_dev_dep\n"
            + "\n"
            + "Not imported, but declared as direct dependencies (may cause the build to fail):\n"
            + "    direct_dep, direct_dev_dep\n"
            + "\n"
            + "Imported, but not declared as direct dependencies:\n"
            + "    indirect_dep, indirect_dev_dep\n"
            + "\n"
            + "\033[35m\033[1m ** You can use the following buildozer command(s) to fix these issues:\033[0m\n"
            + "\n"
            + "buildozer 'use_repo_add @mod//:extensions.bzl ext direct_dep' //MODULE.bazel:all\n"
            + "buildozer 'use_repo_remove @mod//:extensions.bzl ext indirect_dep invalid_dep' //MODULE.bazel:all\n"
            + "buildozer 'use_repo_add dev @mod//:extensions.bzl ext direct_dev_dep' //MODULE.bazel:all\n"
            + "buildozer 'use_repo_remove dev @mod//:extensions.bzl ext indirect_dev_dep invalid_dev_dep' //MODULE.bazel:all");
  }

  @Test
  public void testGenerateFixupMessage_useAllRepos() throws EvalException {
    var moduleExtensionMetadata = ModuleExtensionMetadata.create("all", StarlarkList.immutableOf());
    var fixupMessage = moduleExtensionMetadata.generateFixupMessage(TEST_USAGES, TEST_ALL_REPOS);

    assertThat(fixupMessage.isPresent()).isTrue();
    assertThat(fixupMessage.get().getLocation()).isEqualTo(TEST_ROOT_LOCATION);
    assertThat(fixupMessage.get().getKind()).isEqualTo(EventKind.WARNING);
    assertThat(fixupMessage.get().getMessage()).isEqualTo(
        "The module extension ext defined in @mod//:extensions.bzl reported incorrect imports of repositories via use_repo():\n"
            + "\n"
            + "Imported, but not created by the extension (will cause the build to fail):\n"
            + "    invalid_dep, invalid_dev_dep\n"
            + "\n"
            + "Not imported, but declared as direct dependencies (may cause the build to fail):\n"
            + "    direct_dep, direct_dev_dep\n"
            + "\n"
            + "\033[35m\033[1m ** You can use the following buildozer command(s) to fix these issues:\033[0m\n"
            + "\n"
            + "buildozer 'use_repo_add @mod//:extensions.bzl ext direct_dep direct_dev_dep indirect_dev_dep' //MODULE.bazel:all\n"
            + "buildozer 'use_repo_remove @mod//:extensions.bzl ext invalid_dep' //MODULE.bazel:all\n"
            + "buildozer 'use_repo_remove dev @mod//:extensions.bzl ext indirect_dev_dep invalid_dev_dep' //MODULE.bazel:all");
  }

  @Test
  public void testGenerateFixupMessage_useAllRepos_dev() throws EvalException {
    var moduleExtensionMetadata = ModuleExtensionMetadata.create(Starlark.NONE, "all");
    var fixupMessage = moduleExtensionMetadata.generateFixupMessage(TEST_USAGES, TEST_ALL_REPOS);

    assertThat(fixupMessage.isPresent()).isTrue();
    assertThat(fixupMessage.get().getLocation()).isEqualTo(TEST_ROOT_LOCATION);
    assertThat(fixupMessage.get().getKind()).isEqualTo(EventKind.WARNING);
    assertThat(fixupMessage.get().getMessage()).isEqualTo(
        "The module extension ext defined in @mod//:extensions.bzl reported incorrect imports of repositories via use_repo():\n"
            + "\n"
            + "Imported, but not created by the extension (will cause the build to fail):\n"
            + "    invalid_dep, invalid_dev_dep\n"
            + "\n"
            + "Not imported, but declared as direct dependencies (may cause the build to fail):\n"
            + "    direct_dep, direct_dev_dep\n"
            + "\n"
            + "\033[35m\033[1m ** You can use the following buildozer command(s) to fix these issues:\033[0m\n"
            + "\n"
            + "buildozer 'use_repo_remove @mod//:extensions.bzl ext indirect_dep invalid_dep' //MODULE.bazel:all\n"
            + "buildozer 'use_repo_add dev @mod//:extensions.bzl ext direct_dep direct_dev_dep indirect_dep' //MODULE.bazel:all\n"
            + "buildozer 'use_repo_remove dev @mod//:extensions.bzl ext invalid_dev_dep' //MODULE.bazel:all");
  }
}
