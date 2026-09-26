// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.runtime.CommandLinePathFactory.CommandLinePathFactoryException;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.OutputStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommandLinePathFactory}. */
@RunWith(JUnit4.class)
public class CommandLinePathFactoryTest {
  private static final Joiner PATH_JOINER = Joiner.on(File.pathSeparator);

  private FileSystem filesystem = null;

  @Before
  public void prepareFilesystem() throws Exception {
    filesystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
  }

  private void createExecutable(String path) throws Exception {
    Preconditions.checkNotNull(path);

    createExecutable(filesystem.getPath(path));
  }

  private void createExecutable(Path path) throws Exception {
    Preconditions.checkNotNull(path);

    path.getParentDirectory().createDirectoryAndParents();
    try (OutputStream stream = path.getOutputStream()) {
      // Just create an empty file, nothing to do.
    }
    path.setExecutable(true);
  }

  @Test
  public void emptyPathIsRejected() {
    CommandLinePathFactory factory = new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.LINUX);

    assertThrows(IllegalArgumentException.class, () -> factory.create(ImmutableMap.of(), ""));
  }

  @Test
  public void createFromAbsolutePath() throws Exception {
    CommandLinePathFactory factory = new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.LINUX);

    assertThat(factory.create(ImmutableMap.of(), "/absolute/path/1"))
        .isEqualTo(filesystem.getPath("/absolute/path/1"));
    assertThat(factory.create(ImmutableMap.of(), "/absolute/path/2"))
        .isEqualTo(filesystem.getPath("/absolute/path/2"));
  }

  @Test
  public void createWithNamedRoot() throws Exception {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(
            filesystem,
            ImmutableMap.of(
                "workspace", filesystem.getPath("/path/to/workspace"),
                "output_base", filesystem.getPath("/path/to/output/base")),
            OS.LINUX);

    assertThat(factory.create(ImmutableMap.of(), "/absolute/path/1"))
        .isEqualTo(filesystem.getPath("/absolute/path/1"));
    assertThat(factory.create(ImmutableMap.of(), "/absolute/path/2"))
        .isEqualTo(filesystem.getPath("/absolute/path/2"));

    assertThat(factory.create(ImmutableMap.of(), "%workspace%/foo"))
        .isEqualTo(filesystem.getPath("/path/to/workspace/foo"));
    assertThat(factory.create(ImmutableMap.of(), "%workspace%/foo/bar"))
        .isEqualTo(filesystem.getPath("/path/to/workspace/foo/bar"));

    assertThat(factory.create(ImmutableMap.of(), "%output_base%/foo"))
        .isEqualTo(filesystem.getPath("/path/to/output/base/foo"));
    assertThat(factory.create(ImmutableMap.of(), "%output_base%/foo/bar"))
        .isEqualTo(filesystem.getPath("/path/to/output/base/foo/bar"));

    assertThat(factory.create(ImmutableMap.of(), "%workspace%//foo//bar"))
        .isEqualTo(filesystem.getPath("/path/to/workspace/foo/bar"));
  }

  @Test
  public void pathLeakingOutsideOfRoot() {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(
            filesystem, ImmutableMap.of("a", filesystem.getPath("/path/to/a")), OS.LINUX);

    assertThrows(
        CommandLinePathFactoryException.class,
        () -> factory.create(ImmutableMap.of(), "%a%/../foo"));
    assertThrows(
        CommandLinePathFactoryException.class,
        () -> factory.create(ImmutableMap.of(), "%a%/b/../.."));
  }

  @Test
  public void unknownRoot() {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(
            filesystem, ImmutableMap.of("a", filesystem.getPath("/path/to/a")), OS.LINUX);

    assertThrows(
        CommandLinePathFactoryException.class,
        () -> factory.create(ImmutableMap.of(), "%workspace%/foo"));
    assertThrows(
        CommandLinePathFactoryException.class,
        () -> factory.create(ImmutableMap.of(), "%output_base%/foo"));
  }

  @Test
  public void relativePathWithMultipleSegments() {
    CommandLinePathFactory factory = new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.LINUX);

    assertThrows(
        CommandLinePathFactoryException.class, () -> factory.create(ImmutableMap.of(), "a/b"));
    assertThrows(
        CommandLinePathFactoryException.class, () -> factory.create(ImmutableMap.of(), "a/b/c/d"));
  }

  @Test
  public void pathLookup() throws Exception {
    CommandLinePathFactory factory = new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.LINUX);

    createExecutable("/bin/true");
    createExecutable("/bin/false");
    createExecutable("/usr/bin/foo-bar.exe");
    createExecutable("/usr/local/bin/baz");
    createExecutable("/home/yannic/bin/abc");
    createExecutable("/home/yannic/bin/true");

    var path =
        ImmutableMap.of(
            "PATH", PATH_JOINER.join("/bin", "/usr/bin", "/usr/local/bin", "/home/yannic/bin"));
    assertThat(factory.create(path, "true")).isEqualTo(filesystem.getPath("/bin/true"));
    assertThat(factory.create(path, "false")).isEqualTo(filesystem.getPath("/bin/false"));
    assertThat(factory.create(path, "foo-bar.exe"))
        .isEqualTo(filesystem.getPath("/usr/bin/foo-bar.exe"));
    assertThat(factory.create(path, "baz")).isEqualTo(filesystem.getPath("/usr/local/bin/baz"));
    assertThat(factory.create(path, "abc")).isEqualTo(filesystem.getPath("/home/yannic/bin/abc"));

    // `.exe` is required; extensions are only appended on Windows.
    assertThrows(FileNotFoundException.class, () -> factory.create(path, "foo-bar"));
  }

  // Note: the tests below run against a case-sensitive in-memory filesystem, so the fixtures spell
  // extensions with the exact case used by `PATHEXT`. Real Windows filesystems are
  // case-insensitive, so `foo.exe` is found there for a `PATHEXT` entry of `.EXE` as well.

  @Test
  public void pathLookupOnWindowsAppendsDefaultExtensions() throws Exception {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.WINDOWS);

    createExecutable("/bin/foo.EXE");
    createExecutable("/bin/bar.CMD");
    createExecutable("/bin/baz");

    var path = ImmutableMap.of("PATH", PATH_JOINER.join("/bin"));

    // `PATHEXT` is unset, so the Windows defaults apply.
    assertThat(factory.create(path, "foo")).isEqualTo(filesystem.getPath("/bin/foo.EXE"));
    assertThat(factory.create(path, "bar")).isEqualTo(filesystem.getPath("/bin/bar.CMD"));

    // A name that exists verbatim is still found, and spelling out the extension keeps working.
    assertThat(factory.create(path, "baz")).isEqualTo(filesystem.getPath("/bin/baz"));
    assertThat(factory.create(path, "foo.EXE")).isEqualTo(filesystem.getPath("/bin/foo.EXE"));

    assertThrows(FileNotFoundException.class, () -> factory.create(path, "quux"));
  }

  @Test
  public void pathLookupOnWindowsHonorsPathExt() throws Exception {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.WINDOWS);

    createExecutable("/bin/foo.exe");
    createExecutable("/bin/bar.py");

    var path =
        ImmutableMap.of(
            "PATH", PATH_JOINER.join("/bin"),
            // Also covers an entry without a leading dot, which Windows tolerates.
            "PATHEXT", ".com;.exe;py");

    assertThat(factory.create(path, "foo")).isEqualTo(filesystem.getPath("/bin/foo.exe"));
    assertThat(factory.create(path, "bar")).isEqualTo(filesystem.getPath("/bin/bar.py"));
  }

  @Test
  public void pathLookupOnWindowsWithPathExtNotListingExtension() {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.WINDOWS);

    var path = ImmutableMap.of("PATH", PATH_JOINER.join("/bin"), "PATHEXT", ".com;.exe");

    assertThrows(FileNotFoundException.class, () -> factory.create(path, "foo"));
  }

  @Test
  public void pathLookupOnWindowsPrefersEarlierPathEntry() throws Exception {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.WINDOWS);

    createExecutable("/bin/foo.cmd");
    createExecutable("/usr/bin/foo.exe");

    var path =
        ImmutableMap.of(
            "PATH", PATH_JOINER.join("/bin", "/usr/bin"), "PATHEXT", ".exe;.cmd");

    // The `PATH` order wins over the `PATHEXT` order.
    assertThat(factory.create(path, "foo")).isEqualTo(filesystem.getPath("/bin/foo.cmd"));
  }

  @Test
  public void pathLookupOnWindowsWithEmptyPathExt() throws Exception {
    CommandLinePathFactory factory =
        new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.WINDOWS);

    createExecutable("/bin/foo.EXE");

    // An empty `PATHEXT` falls back to the defaults rather than disabling the lookup.
    var path = ImmutableMap.of("PATH", PATH_JOINER.join("/bin"), "PATHEXT", "");

    assertThat(factory.create(path, "foo")).isEqualTo(filesystem.getPath("/bin/foo.EXE"));
  }

  @Test
  public void pathLookupWithUndefinedPath() {
    CommandLinePathFactory factory = new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.LINUX);

    assertThrows(FileNotFoundException.class, () -> factory.create(ImmutableMap.of(), "a"));
    assertThrows(FileNotFoundException.class, () -> factory.create(ImmutableMap.of(), "foo"));
  }

  @Test
  public void pathLookupWithNonExistingDirectoryOnPath() {
    CommandLinePathFactory factory = new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.LINUX);

    assertThrows(
        FileNotFoundException.class,
        () -> factory.create(ImmutableMap.of("PATH", "/does/not/exist"), "a"));
  }

  @Test
  public void pathLookupWithExistingAndNonExistingDirectoryOnPath() throws Exception {
    CommandLinePathFactory factory = new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.LINUX);

    createExecutable("/bin/foo");
    createExecutable("/usr/bin/bar");
    assertThrows(
        FileNotFoundException.class,
        () ->
            factory.create(
                ImmutableMap.of("PATH", PATH_JOINER.join("/bin", "/does/not/exist", "/usr/bin")),
                "a"));
  }

  @Test
  public void pathLookupWithInvalidPath() throws Exception {
    CommandLinePathFactory factory = new CommandLinePathFactory(filesystem, ImmutableMap.of(), OS.LINUX);

    createExecutable("/bin/true");
    var path = ImmutableMap.of("PATH", PATH_JOINER.join("", ".", "/bin"));
    assertThat(factory.create(path, "true")).isEqualTo(filesystem.getPath("/bin/true"));
  }
}
