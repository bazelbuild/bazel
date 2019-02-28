package com.google.devtools.build.lib.blackbox.tests;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.repository.DecompressorDescriptor;
import com.google.devtools.build.lib.bazel.repository.TarFunction;
import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;

/**
 * TODO
 */
public class BazelEmbeddedSkylarkBlackBoxTest extends AbstractBlackBoxTest {

  public static final String HELLO_FROM_EXTERNAL_REPOSITORY = "Hello from external repository!";
  public static final String HELLO_FROM_MAIN_REPOSITORY = "Hello from main repository!";

  @Test
  public void testPkgTar() throws Exception {
    context().write("main/WORKSPACE");
    context().write("main/foo.txt", "Hello World");
    context().write("main/bar.txt", "Hello World, again");
    context().write("main/BUILD",
        "load(\"@bazel_tools//tools/build_defs/pkg:pkg.bzl\", \"pkg_tar\")",
        "pkg_tar(name = \"data\", srcs = glob([\"*.txt\"]),)");

    BuilderRunner bazel = context().bazel().withFlags("--all_incompatible_changes");
    bazel.build("...");

    Path dataTarPath = context().resolveBinPath(bazel, "main/data.tar");
    assertThat(Files.exists(dataTarPath)).isTrue();

    FileSystem fs = FileSystems.getJavaIoFileSystem();
    com.google.devtools.build.lib.vfs.Path dataTarPathForDecompress =
        fs.getPath(dataTarPath.toAbsolutePath().toString());
    com.google.devtools.build.lib.vfs.Path directory =
        TarFunction.INSTANCE
        .decompress(DecompressorDescriptor
            .builder()
            .setArchivePath(dataTarPathForDecompress)
            .setRepositoryPath(dataTarPathForDecompress.getParentDirectory())
            .build());
    assertThat(directory.exists()).isTrue();
    List<String> children = directory.getDirectoryEntries().stream()
        .map(path -> path.getBaseName()).collect(Collectors.toList());
    assertThat(children).contains("foo.txt");
    assertThat(children).contains("bar.txt");
  }

  @Test
  public void testHttpArchive() throws Exception {
    Path repo = HelperStarlarkTexts
        .setupRepositoryWithRuleWritingTextToFile(context().getTmpDir(), "ext_repo",
            HELLO_FROM_EXTERNAL_REPOSITORY).toAbsolutePath();
    Path build = repo.resolve("BUILD");

    PathUtils.append(build,
        "load(\"@bazel_tools//tools/build_defs/pkg:pkg.bzl\", \"pkg_tar\")",
        "pkg_tar(name = \"packed_ext_repo\", srcs = glob([\"*\"]),)");

    System.out.println("\nWORKSPACE: \n" + String.join("\n", PathUtils.readFile(repo.resolve("WORKSPACE"))) + "\n88888888888\n");
    System.out.println("BUILD: \n" + String.join("\n", PathUtils.readFile(repo.resolve("BUILD"))) + "\n88888888888\n");

    Path zipFile = context().getTmpDir().resolve("ext_repo.tar");
    assertThat(Files.exists(zipFile)).isFalse();

    context().write("WORKSPACE",
        "load(\"@bazel_tools//tools/build_defs/repo:http.bzl\", \"http_archive\")\n",
        String.format("local_repository(name=\"ext_local\", path=\"%s\",)", pathToString(repo)),
        String.format("http_archive(name=\"ext\", urls=[\"file://%s\"],)", zipFile.toString().replace("\\", "/")));

    context().write("BUILD", HelperStarlarkTexts.getLoadForRuleWritingTextToFile("@ext"),
        HelperStarlarkTexts.callRuleWritingTextToFile("call_from_main", "main_out.txt",
            HELLO_FROM_MAIN_REPOSITORY));

    System.out.println("\nMAIN WORKSPACE: \n" + String.join("\n", PathUtils.readFile(context().getWorkDir().resolve("WORKSPACE"))) + "\n88888888888\n");
    System.out.println("MAIN BUILD: \n" + String.join("\n", PathUtils.readFile(context().getWorkDir().resolve("BUILD"))) + "\n88888888888\n");

    // first build the archive and copy it into zipFile
    BuilderRunner bazel = context().bazel();
    bazel.build("@ext_local//:packed_ext_repo");
    Path packedFile = context().resolveBinPath(bazel, "external/ext_local/packed_ext_repo.tar");
    Files.copy(packedFile, zipFile);

    bazel.build("@ext//:x");

    Path xPath = context().resolveBinPath(bazel, "external/ext/out");
    assertThat(Files.exists(xPath)).isTrue();
    List<String> lines = PathUtils.readFile(xPath);
    assertThat(lines.size()).isEqualTo(1);
    assertThat(lines.get(0)).isEqualTo(HELLO_FROM_EXTERNAL_REPOSITORY);

    bazel.build("//:call_from_main");

    Path mainOutPath = context().resolveBinPath(bazel, "main_out.txt");
    assertThat(Files.exists(mainOutPath)).isTrue();
    List<String> mainOutLines = PathUtils.readFile(mainOutPath);
    assertThat(mainOutLines.size()).isEqualTo(1);
    assertThat(mainOutLines.get(0)).isEqualTo(HELLO_FROM_MAIN_REPOSITORY);
  }

  // todo test tar quoting
}
