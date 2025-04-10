package com.google.devtools.build.lib.actions.cache;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.windows.WindowsFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public class VirtualActionInputTest {
  @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

  public enum FileSystemType {
    IN_MEMORY,
    JAVA,
    NATIVE;

    FileSystem getFileSystem() {
      return switch (this) {
        case IN_MEMORY -> new InMemoryFileSystem(DigestHashFunction.SHA256);
        case JAVA -> new JavaIoFileSystem(DigestHashFunction.SHA256);
        case NATIVE ->
            OS.getCurrent() == OS.WINDOWS
                ? new WindowsFileSystem(DigestHashFunction.SHA256, /* createSymbolicLinks= */ false)
                : new UnixFileSystem(DigestHashFunction.SHA256, "hash");
      };
    }
  }

  @Test
  public void testAtomicallyWriteRelativeTo(@TestParameter FileSystemType fileSystemType)
      throws Exception {
    FileSystem fs = fileSystemType.getFileSystem();
    Path execRoot = fs.getPath(tempFolder.getRoot().getPath());

    Path outputFile = execRoot.getRelative("some/file");
    VirtualActionInput input =
        ActionsTestUtil.createVirtualActionInput(
            outputFile.relativeTo(execRoot).getPathString(), "hello");

    input.atomicallyWriteRelativeTo(execRoot);

    assertThat(outputFile.getParentDirectory().readdir(Symlinks.NOFOLLOW))
        .containsExactly(new Dirent("file", Dirent.Type.FILE));
    assertThat(FileSystemUtils.readLines(outputFile, UTF_8)).containsExactly("hello");
    assertThat(outputFile.isExecutable()).isTrue();

    // Verify that the write succeeds even with concurrent read access to the file.
    try (var in = outputFile.getInputStream()) {
      input.atomicallyWriteRelativeTo(execRoot);
      assertThat(in.readAllBytes()).isEqualTo("hello".getBytes(UTF_8));
    }
  }
}
