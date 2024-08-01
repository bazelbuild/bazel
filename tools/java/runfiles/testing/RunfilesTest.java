// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.runfiles;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Runfiles}. */
@RunWith(JUnit4.class)
public final class RunfilesTest {

  @Rule
  public TemporaryFolder tempDir = new TemporaryFolder(new File(System.getenv("TEST_TMPDIR")));

  private static boolean isWindows() {
    return File.separatorChar == '\\';
  }

  private void assertRlocationArg(Runfiles runfiles, String path, @Nullable String error) {
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> runfiles.rlocation(path));
    if (error != null) {
      assertThat(e).hasMessageThat().contains(error);
    }
  }

  @Test
  public void testRlocationArgumentValidation() throws Exception {
    Path dir =
        Files.createTempDirectory(
            FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR")), null);

    Runfiles r = Runfiles.create(ImmutableMap.of("RUNFILES_DIR", dir.toString()));
    assertRlocationArg(r, null, null);
    assertRlocationArg(r, "", null);
    assertRlocationArg(r, "../foo", "is not normalized");
    assertRlocationArg(r, "foo/..", "is not normalized");
    assertRlocationArg(r, "foo/../bar", "is not normalized");
    assertRlocationArg(r, "./foo", "is not normalized");
    assertRlocationArg(r, "foo/.", "is not normalized");
    assertRlocationArg(r, "foo/./bar", "is not normalized");
    assertRlocationArg(r, "//foobar", "is not normalized");
    assertRlocationArg(r, "foo//", "is not normalized");
    assertRlocationArg(r, "foo//bar", "is not normalized");
    assertRlocationArg(r, "\\foo", "path is absolute without a drive letter");
  }

  @Test
  public void testCreatesManifestBasedRunfiles() throws Exception {
    Path mf = tempFile("foo.runfiles_manifest", ImmutableList.of("a/b c/d"));
    Runfiles r =
        Runfiles.create(
            ImmutableMap.of(
                "RUNFILES_MANIFEST_ONLY", "1",
                "RUNFILES_MANIFEST_FILE", mf.toString(),
                "RUNFILES_DIR", "ignored when RUNFILES_MANIFEST_ONLY=1",
                "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                "TEST_SRCDIR", "should always be ignored"));
    assertThat(r.rlocation("a/b")).isEqualTo("c/d");
    assertThat(r.rlocation("foo")).isNull();

    if (isWindows()) {
      assertThat(r.rlocation("c:/foo")).isEqualTo("c:/foo");
      assertThat(r.rlocation("c:\\foo")).isEqualTo("c:\\foo");
    } else {
      assertThat(r.rlocation("/foo")).isEqualTo("/foo");
    }
  }

  @Test
  public void testCreatesDirectoryBasedRunfiles() throws Exception {
    Path dir =
        Files.createTempDirectory(
            FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR")), null);

    Runfiles r =
        Runfiles.create(
            ImmutableMap.of(
                "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                "RUNFILES_DIR", dir.toString(),
                "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                "TEST_SRCDIR", "should always be ignored"));
    assertThat(r.rlocation("a/b")).endsWith("/a/b");
    assertThat(r.rlocation("foo")).endsWith("/foo");

    r =
        Runfiles.create(
            ImmutableMap.of(
                "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                "RUNFILES_DIR", "",
                "JAVA_RUNFILES", dir.toString(),
                "TEST_SRCDIR", "should always be ignored"));
    assertThat(r.rlocation("a/b")).endsWith("/a/b");
    assertThat(r.rlocation("foo")).endsWith("/foo");
  }

  @Test
  public void testIgnoresTestSrcdirWhenJavaRunfilesIsUndefinedAndJustFails() throws Exception {
    Path dir =
        Files.createTempDirectory(
            FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR")), null);

    Runfiles.create(
        ImmutableMap.of(
            "RUNFILES_DIR", dir.toString(),
            "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
            "TEST_SRCDIR", "should always be ignored"));

    Runfiles.create(
        ImmutableMap.of(
            "JAVA_RUNFILES", dir.toString(),
            "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
            "TEST_SRCDIR", "should always be ignored"));

    IOException e =
        assertThrows(
            IOException.class,
            () ->
                Runfiles.create(
                    ImmutableMap.of(
                        "RUNFILES_DIR",
                        "",
                        "JAVA_RUNFILES",
                        "",
                        "RUNFILES_MANIFEST_FILE",
                        "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                        "TEST_SRCDIR",
                        "should always be ignored")));
    assertThat(e).hasMessageThat().contains("$RUNFILES_DIR and $JAVA_RUNFILES");
  }

  @Test
  public void testFailsToCreateManifestBasedBecauseManifestDoesNotExist() {
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                Runfiles.create(
                    ImmutableMap.of(
                        "RUNFILES_MANIFEST_ONLY", "1",
                        "RUNFILES_MANIFEST_FILE", "non-existing path")));
    assertThat(e).hasMessageThat().contains("non-existing path");
  }

  @Test
  public void testManifestBasedEnvVars() throws Exception {
    Path mf = tempFile("MANIFEST", ImmutableList.of());
    Map<String, String> envvars =
        Runfiles.create(
                ImmutableMap.of(
                    "RUNFILES_MANIFEST_ONLY", "1",
                    "RUNFILES_MANIFEST_FILE", mf.toString(),
                    "RUNFILES_DIR", "ignored when RUNFILES_MANIFEST_ONLY=1",
                    "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                    "TEST_SRCDIR", "should always be ignored"))
            .getEnvVars();
    assertThat(envvars.keySet())
        .containsExactly(
            "RUNFILES_MANIFEST_ONLY", "RUNFILES_MANIFEST_FILE", "RUNFILES_DIR", "JAVA_RUNFILES");
    assertThat(envvars.get("RUNFILES_MANIFEST_ONLY")).isEqualTo("1");
    assertThat(envvars.get("RUNFILES_MANIFEST_FILE")).isEqualTo(mf.toString());
    assertThat(envvars.get("RUNFILES_DIR")).isEqualTo(tempDir.getRoot().toString());
    assertThat(envvars.get("JAVA_RUNFILES")).isEqualTo(tempDir.getRoot().toString());

    Path rfDir = tempDir.getRoot().toPath().resolve("foo.runfiles");
    Files.createDirectories(rfDir);
    mf = tempFile("foo.runfiles_manifest", ImmutableList.of());
    envvars =
        Runfiles.create(
                ImmutableMap.of(
                    "RUNFILES_MANIFEST_ONLY", "1",
                    "RUNFILES_MANIFEST_FILE", mf.toString(),
                    "RUNFILES_DIR", "ignored when RUNFILES_MANIFEST_ONLY=1",
                    "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                    "TEST_SRCDIR", "should always be ignored"))
            .getEnvVars();
    assertThat(envvars.get("RUNFILES_MANIFEST_ONLY")).isEqualTo("1");
    assertThat(envvars.get("RUNFILES_MANIFEST_FILE")).isEqualTo(mf.toString());
    assertThat(envvars.get("RUNFILES_DIR")).isEqualTo(rfDir.toString());
    assertThat(envvars.get("JAVA_RUNFILES")).isEqualTo(rfDir.toString());
  }

  @Test
  public void testDirectoryBasedEnvVars() throws Exception {
    Map<String, String> envvars =
        Runfiles.create(
                ImmutableMap.of(
                    "RUNFILES_MANIFEST_FILE",
                    "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                    "RUNFILES_DIR",
                    tempDir.getRoot().toString(),
                    "JAVA_RUNFILES",
                    "ignored when RUNFILES_DIR has a value",
                    "TEST_SRCDIR",
                    "should always be ignored"))
            .getEnvVars();
    assertThat(envvars.keySet()).containsExactly("RUNFILES_DIR", "JAVA_RUNFILES");
    assertThat(envvars.get("RUNFILES_DIR")).isEqualTo(tempDir.getRoot().toString());
    assertThat(envvars.get("JAVA_RUNFILES")).isEqualTo(tempDir.getRoot().toString());
  }

  @Test
  public void testDirectoryBasedRlocation() throws IOException {
    // The DirectoryBased implementation simply joins the runfiles directory and the runfile's path
    // on a "/". DirectoryBased does not perform any normalization, nor does it check that the path
    // exists.
    File dir = new File(System.getenv("TEST_TMPDIR"), "mock/runfiles");
    assertThat(dir.mkdirs()).isTrue();
    Runfiles r = Runfiles.createDirectoryBasedForTesting(dir.toString()).withSourceRepository("");
    // Escaping for "\": once for string and once for regex.
    assertThat(r.rlocation("arg")).matches(".*[/\\\\]mock[/\\\\]runfiles[/\\\\]arg");
  }

  @Test
  public void testManifestBasedRlocation() throws Exception {
    Path mf =
        tempFile(
            "MANIFEST",
            ImmutableList.of(
                "Foo/runfile1 C:/Actual Path\\runfile1",
                "Foo/Bar/runfile2 D:\\the path\\run file 2.txt",
                "Foo/Bar/Dir E:\\Actual Path\\Directory"));
    Runfiles r = Runfiles.createManifestBasedForTesting(mf.toString()).withSourceRepository("");
    assertThat(r.rlocation("Foo/runfile1")).isEqualTo("C:/Actual Path\\runfile1");
    assertThat(r.rlocation("Foo/Bar/runfile2")).isEqualTo("D:\\the path\\run file 2.txt");
    assertThat(r.rlocation("Foo/Bar/Dir")).isEqualTo("E:\\Actual Path\\Directory");
    assertThat(r.rlocation("Foo/Bar/Dir/File")).isEqualTo("E:\\Actual Path\\Directory/File");
    assertThat(r.rlocation("Foo/Bar/Dir/Deeply/Nested/File"))
        .isEqualTo("E:\\Actual Path\\Directory/Deeply/Nested/File");
    assertThat(r.rlocation("unknown")).isNull();
  }

  @Test
  public void testManifestBasedRlocationWithRepoMapping_fromMain() throws Exception {
    Path rm =
        tempFile(
            "foo.repo_mapping",
            ImmutableList.of(
                ",config.json,config.json+1.2.3",
                ",my_module,_main",
                ",my_protobuf,protobuf+3.19.2",
                ",my_workspace,_main",
                "protobuf+3.19.2,config.json,config.json+1.2.3",
                "protobuf+3.19.2,protobuf,protobuf+3.19.2"));
    Path mf =
        tempFile(
            "foo.runfiles_manifest",
            ImmutableList.of(
                "_repo_mapping " + rm,
                "config.json /etc/config.json",
                "protobuf+3.19.2/foo/runfile C:/Actual Path\\protobuf\\runfile",
                "_main/bar/runfile /the/path/./to/other//other runfile.txt",
                "protobuf+3.19.2/bar/dir E:\\Actual Path\\Directory"));
    Runfiles r = Runfiles.createManifestBasedForTesting(mf.toString()).withSourceRepository("");

    assertThat(r.rlocation("my_module/bar/runfile"))
        .isEqualTo("/the/path/./to/other//other runfile.txt");
    assertThat(r.rlocation("my_workspace/bar/runfile"))
        .isEqualTo("/the/path/./to/other//other runfile.txt");
    assertThat(r.rlocation("my_protobuf/foo/runfile"))
        .isEqualTo("C:/Actual Path\\protobuf\\runfile");
    assertThat(r.rlocation("my_protobuf/bar/dir")).isEqualTo("E:\\Actual Path\\Directory");
    assertThat(r.rlocation("my_protobuf/bar/dir/file"))
        .isEqualTo("E:\\Actual Path\\Directory/file");
    assertThat(r.rlocation("my_protobuf/bar/dir/de eply/nes ted/fi+le"))
        .isEqualTo("E:\\Actual Path\\Directory/de eply/nes ted/fi+le");

    assertThat(r.rlocation("protobuf/foo/runfile")).isNull();
    assertThat(r.rlocation("protobuf/bar/dir")).isNull();
    assertThat(r.rlocation("protobuf/bar/dir/file")).isNull();
    assertThat(r.rlocation("protobuf/bar/dir/dir/de eply/nes ted/fi+le")).isNull();

    assertThat(r.rlocation("_main/bar/runfile"))
        .isEqualTo("/the/path/./to/other//other runfile.txt");
    assertThat(r.rlocation("protobuf+3.19.2/foo/runfile"))
        .isEqualTo("C:/Actual Path\\protobuf\\runfile");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir")).isEqualTo("E:\\Actual Path\\Directory");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/file"))
        .isEqualTo("E:\\Actual Path\\Directory/file");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo("E:\\Actual Path\\Directory/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("config.json")).isEqualTo("/etc/config.json");
    assertThat(r.rlocation("_main")).isNull();
    assertThat(r.rlocation("my_module")).isNull();
    assertThat(r.rlocation("protobuf")).isNull();
  }

  @Test
  public void testManifestBasedRlocationUnmapped() throws Exception {
    Path rm =
        tempFile(
            "foo.repo_mapping",
            ImmutableList.of(
                ",config.json,config.json+1.2.3",
                ",my_module,_main",
                ",my_protobuf,protobuf+3.19.2",
                ",my_workspace,_main",
                "protobuf+3.19.2,config.json,config.json+1.2.3",
                "protobuf+3.19.2,protobuf,protobuf+3.19.2"));
    Path mf =
        tempFile(
            "foo.runfiles_manifest",
            ImmutableList.of(
                "_repo_mapping " + rm,
                "config.json /etc/config.json",
                "protobuf+3.19.2/foo/runfile C:/Actual Path\\protobuf\\runfile",
                "_main/bar/runfile /the/path/./to/other//other runfile.txt",
                "protobuf+3.19.2/bar/dir E:\\Actual Path\\Directory"));
    Runfiles r = Runfiles.createManifestBasedForTesting(mf.toString()).unmapped();

    assertThat(r.rlocation("my_module/bar/runfile")).isNull();
    assertThat(r.rlocation("my_workspace/bar/runfile")).isNull();
    assertThat(r.rlocation("my_protobuf/foo/runfile")).isNull();
    assertThat(r.rlocation("my_protobuf/bar/dir")).isNull();
    assertThat(r.rlocation("my_protobuf/bar/dir/file")).isNull();
    assertThat(r.rlocation("my_protobuf/bar/dir/de eply/nes ted/fi+le")).isNull();

    assertThat(r.rlocation("protobuf/foo/runfile")).isNull();
    assertThat(r.rlocation("protobuf/bar/dir")).isNull();
    assertThat(r.rlocation("protobuf/bar/dir/file")).isNull();
    assertThat(r.rlocation("protobuf/bar/dir/dir/de eply/nes ted/fi+le")).isNull();

    assertThat(r.rlocation("_main/bar/runfile"))
        .isEqualTo("/the/path/./to/other//other runfile.txt");
    assertThat(r.rlocation("protobuf+3.19.2/foo/runfile"))
        .isEqualTo("C:/Actual Path\\protobuf\\runfile");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir")).isEqualTo("E:\\Actual Path\\Directory");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/file"))
        .isEqualTo("E:\\Actual Path\\Directory/file");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo("E:\\Actual Path\\Directory/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("config.json")).isEqualTo("/etc/config.json");
    assertThat(r.rlocation("_main")).isNull();
    assertThat(r.rlocation("my_module")).isNull();
    assertThat(r.rlocation("protobuf")).isNull();
  }

  @Test
  public void testManifestBasedRlocationWithRepoMapping_fromOtherRepo() throws Exception {
    Path rm =
        tempFile(
            "foo.repo_mapping",
            ImmutableList.of(
                ",config.json,config.json+1.2.3",
                ",my_module,_main",
                ",my_protobuf,protobuf+3.19.2",
                ",my_workspace,_main",
                "protobuf+3.19.2,config.json,config.json+1.2.3",
                "protobuf+3.19.2,protobuf,protobuf+3.19.2"));
    Path mf =
        tempFile(
            "foo.runfiles/MANIFEST",
            ImmutableList.of(
                "_repo_mapping " + rm,
                "config.json /etc/config.json",
                "protobuf+3.19.2/foo/runfile C:/Actual Path\\protobuf\\runfile",
                "_main/bar/runfile /the/path/./to/other//other runfile.txt",
                "protobuf+3.19.2/bar/dir E:\\Actual Path\\Directory"));
    Runfiles r =
        Runfiles.createManifestBasedForTesting(mf.toString())
            .withSourceRepository("protobuf+3.19.2");

    assertThat(r.rlocation("protobuf/foo/runfile")).isEqualTo("C:/Actual Path\\protobuf\\runfile");
    assertThat(r.rlocation("protobuf/bar/dir")).isEqualTo("E:\\Actual Path\\Directory");
    assertThat(r.rlocation("protobuf/bar/dir/file")).isEqualTo("E:\\Actual Path\\Directory/file");
    assertThat(r.rlocation("protobuf/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo("E:\\Actual Path\\Directory/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("my_module/bar/runfile")).isNull();
    assertThat(r.rlocation("my_protobuf/foo/runfile")).isNull();
    assertThat(r.rlocation("my_protobuf/bar/dir")).isNull();
    assertThat(r.rlocation("my_protobuf/bar/dir/file")).isNull();
    assertThat(r.rlocation("my_protobuf/bar/dir/de eply/nes  ted/fi+le")).isNull();

    assertThat(r.rlocation("_main/bar/runfile"))
        .isEqualTo("/the/path/./to/other//other runfile.txt");
    assertThat(r.rlocation("protobuf+3.19.2/foo/runfile"))
        .isEqualTo("C:/Actual Path\\protobuf\\runfile");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir")).isEqualTo("E:\\Actual Path\\Directory");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/file"))
        .isEqualTo("E:\\Actual Path\\Directory/file");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo("E:\\Actual Path\\Directory/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("config.json")).isEqualTo("/etc/config.json");
    assertThat(r.rlocation("_main")).isNull();
    assertThat(r.rlocation("my_module")).isNull();
    assertThat(r.rlocation("protobuf")).isNull();
  }

  @Test
  public void testDirectoryBasedRlocationWithRepoMapping_fromMain() throws Exception {
    Path dir = tempDir.newFolder("foo.runfiles").toPath();
    Path unused =
        tempFile(
            dir.resolve("_repo_mapping").toString(),
            ImmutableList.of(
                ",config.json,config.json+1.2.3",
                ",my_module,_main",
                ",my_protobuf,protobuf+3.19.2",
                ",my_workspace,_main",
                "protobuf+3.19.2,config.json,config.json+1.2.3",
                "protobuf+3.19.2,protobuf,protobuf+3.19.2"));
    Runfiles r = Runfiles.createDirectoryBasedForTesting(dir.toString()).withSourceRepository("");

    assertThat(r.rlocation("my_module/bar/runfile")).isEqualTo(dir + "/_main/bar/runfile");
    assertThat(r.rlocation("my_workspace/bar/runfile")).isEqualTo(dir + "/_main/bar/runfile");
    assertThat(r.rlocation("my_protobuf/foo/runfile"))
        .isEqualTo(dir + "/protobuf+3.19.2/foo/runfile");
    assertThat(r.rlocation("my_protobuf/bar/dir")).isEqualTo(dir + "/protobuf+3.19.2/bar/dir");
    assertThat(r.rlocation("my_protobuf/bar/dir/file"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/file");
    assertThat(r.rlocation("my_protobuf/bar/dir/de eply/nes ted/fi+le"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/de eply/nes ted/fi+le");

    assertThat(r.rlocation("protobuf/foo/runfile")).isEqualTo(dir + "/protobuf/foo/runfile");
    assertThat(r.rlocation("protobuf/bar/dir/dir/de eply/nes ted/fi+le"))
        .isEqualTo(dir + "/protobuf/bar/dir/dir/de eply/nes ted/fi+le");

    assertThat(r.rlocation("_main/bar/runfile")).isEqualTo(dir + "/_main/bar/runfile");
    assertThat(r.rlocation("protobuf+3.19.2/foo/runfile"))
        .isEqualTo(dir + "/protobuf+3.19.2/foo/runfile");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir")).isEqualTo(dir + "/protobuf+3.19.2/bar/dir");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/file"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/file");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("config.json")).isEqualTo(dir + "/config.json");
  }

  @Test
  public void testDirectoryBasedRlocationUnmapped() throws Exception {
    Path dir = tempDir.newFolder("foo.runfiles").toPath();
    Path unused =
        tempFile(
            dir.resolve("_repo_mapping").toString(),
            ImmutableList.of(
                ",config.json,config.json+1.2.3",
                ",my_module,_main",
                ",my_protobuf,protobuf+3.19.2",
                ",my_workspace,_main",
                "protobuf+3.19.2,config.json,config.json+1.2.3",
                "protobuf+3.19.2,protobuf,protobuf+3.19.2"));
    Runfiles r = Runfiles.createDirectoryBasedForTesting(dir.toString()).unmapped();

    assertThat(r.rlocation("my_module/bar/runfile")).isEqualTo(dir + "/my_module/bar/runfile");
    assertThat(r.rlocation("my_workspace/bar/runfile"))
        .isEqualTo(dir + "/my_workspace/bar/runfile");
    assertThat(r.rlocation("my_protobuf/foo/runfile")).isEqualTo(dir + "/my_protobuf/foo/runfile");
    assertThat(r.rlocation("my_protobuf/bar/dir")).isEqualTo(dir + "/my_protobuf/bar/dir");
    assertThat(r.rlocation("my_protobuf/bar/dir/file"))
        .isEqualTo(dir + "/my_protobuf/bar/dir/file");
    assertThat(r.rlocation("my_protobuf/bar/dir/de eply/nes ted/fi+le"))
        .isEqualTo(dir + "/my_protobuf/bar/dir/de eply/nes ted/fi+le");

    assertThat(r.rlocation("protobuf/foo/runfile")).isEqualTo(dir + "/protobuf/foo/runfile");
    assertThat(r.rlocation("protobuf/bar/dir/dir/de eply/nes ted/fi+le"))
        .isEqualTo(dir + "/protobuf/bar/dir/dir/de eply/nes ted/fi+le");

    assertThat(r.rlocation("_main/bar/runfile")).isEqualTo(dir + "/_main/bar/runfile");
    assertThat(r.rlocation("protobuf+3.19.2/foo/runfile"))
        .isEqualTo(dir + "/protobuf+3.19.2/foo/runfile");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir")).isEqualTo(dir + "/protobuf+3.19.2/bar/dir");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/file"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/file");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("config.json")).isEqualTo(dir + "/config.json");
  }

  @Test
  public void testDirectoryBasedRlocationWithRepoMapping_fromOtherRepo() throws Exception {
    Path dir = tempDir.newFolder("foo.runfiles").toPath();
    Path unused =
        tempFile(
            dir.resolve("_repo_mapping").toString(),
            ImmutableList.of(
                ",config.json,config.json+1.2.3",
                ",my_module,_main",
                ",my_protobuf,protobuf+3.19.2",
                ",my_workspace,_main",
                "protobuf+3.19.2,config.json,config.json+1.2.3",
                "protobuf+3.19.2,protobuf,protobuf+3.19.2"));
    Runfiles r =
        Runfiles.createDirectoryBasedForTesting(dir.toString())
            .withSourceRepository("protobuf+3.19.2");

    assertThat(r.rlocation("protobuf/foo/runfile")).isEqualTo(dir + "/protobuf+3.19.2/foo/runfile");
    assertThat(r.rlocation("protobuf/bar/dir")).isEqualTo(dir + "/protobuf+3.19.2/bar/dir");
    assertThat(r.rlocation("protobuf/bar/dir/file"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/file");
    assertThat(r.rlocation("protobuf/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("my_module/bar/runfile")).isEqualTo(dir + "/my_module/bar/runfile");
    assertThat(r.rlocation("my_protobuf/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo(dir + "/my_protobuf/bar/dir/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("_main/bar/runfile")).isEqualTo(dir + "/_main/bar/runfile");
    assertThat(r.rlocation("protobuf+3.19.2/foo/runfile"))
        .isEqualTo(dir + "/protobuf+3.19.2/foo/runfile");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir")).isEqualTo(dir + "/protobuf+3.19.2/bar/dir");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/file"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/file");
    assertThat(r.rlocation("protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le"))
        .isEqualTo(dir + "/protobuf+3.19.2/bar/dir/de eply/nes  ted/fi+le");

    assertThat(r.rlocation("config.json")).isEqualTo(dir + "/config.json");
  }

  @Test
  public void testDirectoryBasedCtorArgumentValidation() throws IOException {
    assertThrows(
        IllegalArgumentException.class,
        () -> Runfiles.createDirectoryBasedForTesting(null).withSourceRepository(""));

    assertThrows(
        IllegalArgumentException.class,
        () -> Runfiles.createDirectoryBasedForTesting("").withSourceRepository(""));

    assertThrows(
        IllegalArgumentException.class,
        () ->
            Runfiles.createDirectoryBasedForTesting("non-existent directory is bad")
                .withSourceRepository(""));

    var unused =
        Runfiles.createDirectoryBasedForTesting(System.getenv("TEST_TMPDIR"))
            .withSourceRepository("");
  }

  @Test
  public void testManifestBasedCtorArgumentValidation() throws Exception {
    assertThrows(
        IllegalArgumentException.class,
        () -> Runfiles.createManifestBasedForTesting(null).withSourceRepository(""));

    assertThrows(
        IllegalArgumentException.class,
        () -> Runfiles.createManifestBasedForTesting("").withSourceRepository(""));

    Path mf = tempFile("foobar", ImmutableList.of("a b"));
    var unused = Runfiles.createManifestBasedForTesting(mf.toString()).withSourceRepository("");
  }

  @Test
  public void testInvalidRepoMapping() throws Exception {
    Path rm = tempFile("foo.repo_mapping", ImmutableList.of("a,b,c,d"));
    Path mf = tempFile("foo.runfiles/MANIFEST", ImmutableList.of("_repo_mapping " + rm));
    assertThrows(
        IllegalArgumentException.class,
        () -> Runfiles.createManifestBasedForTesting(mf.toString()).withSourceRepository(""));
  }

  private Path tempFile(String path, ImmutableList<String> lines) throws IOException {
    Path file = tempDir.getRoot().toPath().resolve(path.replace('/', File.separatorChar));
    Files.createDirectories(file.getParent());
    return Files.write(file, lines, StandardCharsets.UTF_8);
  }
}
