// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.starlark;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Base64;
import java.util.HexFormat;
import java.util.Map.Entry;
import java.util.concurrent.CompletableFuture;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.SymbolGenerator;
import org.apache.commons.lang3.text.WordUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;

@RunWith(JUnit4.class)
public class StarlarkBaseExternalContextTest {

  /** The sha256 of an empty file (<code>sha256sum /dev/null</code>). */
  public static final String SHA256_EMPTY_FILE =
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

  /**
   * Byte array of an empty gzipped file. You can generate the same sequence of bytes with (sed part
   * is linux-specific):
   *
   * <pre>{@code
   * $ gzip -n < /dev/null > empty.gz
   * $ od -v -t d1 empty.gz | cut -c9- | sed 's/\([0-9]\+\)/\1,/g'
   * }</pre>
   */
  private static final byte[] emptyTarGzBytes =
      new byte[] {31, -117, 8, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  /** Returns a copy of the empty tar.gz bytes. */
  public static byte[] getEmptyTarGzBytes() {
    return emptyTarGzBytes.clone();
  }

  /** The sha256 of a gzipped, empty file ({@code gzip -n < /dev/null | sha256sum}). */
  public static final String SHA256_EMPTY_GZ_FILE =
      "59869db34853933b239f1e2219cf7d431da006aa919635478511fabbfc8849d2";

  @Rule public final MockitoRule mockito = MockitoJUnit.rule();

  private BlazeDirectories blazeDirectories;
  private ProcessWrapper processWrapper;
  @Mock private Environment environment;
  @Mock private DownloadManager downloadManager;
  @Mock private StarlarkSemantics starlarkSemantics;
  @Mock private RepositoryRemoteExecutor remoteExecutor;
  private final Mutability mu = Mutability.create("test");
  StarlarkThread starlarkThread =
      StarlarkThread.create(
          mu,
          StarlarkSemantics.DEFAULT,
          /* contextDescription= */ "",
          SymbolGenerator.create("test"));
  @Mock private ExtendedEventHandler extendedEventHandler;

  /** A concrete class for testing the abstract {@link StarlarkBaseExternalContext}. */
  static class TestStarlarkBaseExternalContext extends StarlarkBaseExternalContext {
    TestStarlarkBaseExternalContext(
        Path workingDirectory,
        BlazeDirectories directories,
        Environment env,
        ImmutableMap<String, String> repoEnv,
        ImmutableMap<String, String> nonstrictRepoEnv,
        DownloadManager downloadManager,
        double timeoutScaling,
        @Nullable ProcessWrapper processWrapper,
        StarlarkSemantics starlarkSemantics,
        String identifyingStringForLogging,
        @Nullable RepositoryRemoteExecutor remoteExecutor,
        boolean allowWatchingPathsOutsideWorkspace) {
      super(
          workingDirectory,
          directories,
          env,
          repoEnv,
          nonstrictRepoEnv,
          downloadManager,
          timeoutScaling,
          processWrapper,
          starlarkSemantics,
          identifyingStringForLogging,
          remoteExecutor,
          allowWatchingPathsOutsideWorkspace);
    }

    @Override
    protected boolean shouldDeleteWorkingDirectoryOnClose(boolean successful) {
      return false;
    }

    @Override
    public boolean isRemotable() {
      return true;
    }

    @Override
    protected ImmutableMap<String, String> getRemoteExecProperties() throws EvalException {
      return ImmutableMap.of();
    }
  }

  /** Calls the given starlark method name on the given object via Java reflection. */
  private static Object callStarlarkMethod(
      Object instance, String starlarkMethodName, Object... args)
      throws InvocationTargetException, IllegalAccessException {
    ImmutableMap<Method, StarlarkMethod> methodAnnotations =
        Starlark.getMethodAnnotations(instance.getClass());
    for (Entry<Method, StarlarkMethod> entry : methodAnnotations.entrySet()) {
      Method javaMethod = entry.getKey();
      StarlarkMethod starlarkMethod = entry.getValue();
      if (starlarkMethodName.equals(starlarkMethod.name())) {
        return javaMethod.invoke(instance, args);
      }
    }
    throw new IllegalArgumentException(
        String.format("Couldn't find the starlark method %s on %s", starlarkMethodName, instance));
  }

  /** Creates a StarlarkContext with the given path. */
  private TestStarlarkBaseExternalContext setupStarlarkContext(Path testPath) {
    return new TestStarlarkBaseExternalContext(
        /* workingDirectory= */ testPath,
        /* directories= */ blazeDirectories,
        /* env= */ environment,
        /* repoEnv= */ null,
        /* nonstrictRepoEnv= */ null,
        /* downloadManager= */ downloadManager,
        /* timeoutScaling= */ 1.0,
        /* processWrapper= */ processWrapper,
        /* starlarkSemantics= */ starlarkSemantics,
        /* identifyingStringForLogging= */ "test",
        /* remoteExecutor= */ remoteExecutor,
        /* allowWatchingPathsOutsideWorkspace= */ false);
  }

  /**
   * Tests the debug printing of <code>repository_ctx.download()</code> when <code>block</code> is
   * <code>false</code> and <code>allow_fail</code> is <code>true</code>.
   */
  @Test
  public void download_asyncAllowFail_debugPrint() throws Exception {
    // Setup a Starlark Context for testing.
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path testPath = fs.getPath("/test");
    testPath.createDirectory();
    testPath.getRelative("output").createDirectory();
    try (StarlarkBaseExternalContext sbec = setupStarlarkContext(testPath)) {
      when(environment.getListener()).thenReturn(extendedEventHandler);
      CompletableFuture<Path> testFuture = new CompletableFuture<>();

      // Fake out the Future (which controls what debugPrint will print).
      when(downloadManager.startDownload(
              /* executorService= */ any(),
              /* originalUrls= */ any(),
              /* headers= */ any(),
              /* authHeaders= */ any(),
              /* checksum= */ any(),
              /* canonicalId= */ any(),
              /* type= */ any(),
              /* output= */ any(),
              /* clientEnv= */ any(),
              /* context= */ any(),
              /* downloadPhaser= */ any(),
              /* mayHardlink= */ anyBoolean()))
          .thenReturn(testFuture);

      Object download =
          sbec.download(
              /* url= */ "http://example.com/file.txt",
              /* output= */ "/test/output",
              /* sha256= */ SHA256_EMPTY_FILE,
              /* executable= */ false,
              /* allowFail= */ true,
              /* canonicalId= */ "id",
              /* authUnchecked= */ Dict.<String, Dict<String, Object>>builder().buildImmutable(),
              /* headersUnchecked= */ Dict.<String, Dict<String, Object>>builder().buildImmutable(),
              /* integrity= */ "",
              /* block= */ false,
              /* thread= */ starlarkThread);
      assertThat(download).isInstanceOf(StarlarkValue.class);
      StarlarkValue starlarkPendingDownload = (StarlarkValue) download;

      // Check that debugPrint shows the RUNNING state.
      Printer runningPrinter = new Printer();
      starlarkPendingDownload.debugPrint(runningPrinter, starlarkThread);
      assertThat(runningPrinter.toString()).contains("(state: RUNNING)");

      // Complete the future and check debugPrint shows the SUCCESS state.
      testFuture.complete(fs.getPath("/test/output"));
      Printer successPrinter = new Printer();
      starlarkPendingDownload.debugPrint(successPrinter, starlarkThread);
      assertThat(successPrinter.toString()).contains("(state: SUCCESS)");

      // Create a download that will fail.
      CompletableFuture<Path> failingFuture = new CompletableFuture<>();
      when(downloadManager.startDownload(
              /* executorService= */ any(),
              /* originalUrls= */ any(),
              /* headers= */ any(),
              /* authHeaders= */ any(),
              /* checksum= */ any(),
              /* canonicalId= */ any(),
              /* type= */ any(),
              /* output= */ any(),
              /* clientEnv= */ any(),
              /* context= */ any(),
              /* downloadPhaser= */ any(),
              /* mayHardlink= */ anyBoolean()))
          .thenReturn(failingFuture);
      Object failingDownload =
          sbec.download(
              /* url= */ "http://example.com/file.txt",
              /* output= */ "/test/output",
              /* sha256= */ SHA256_EMPTY_FILE,
              /* executable= */ false,
              /* allowFail= */ true,
              /* canonicalId= */ "id",
              /* authUnchecked= */ Dict.<String, Dict<String, Object>>builder().buildImmutable(),
              /* headersUnchecked= */ Dict.<String, Dict<String, Object>>builder().buildImmutable(),
              /* integrity= */ "",
              /* block= */ false,
              /* thread= */ starlarkThread);
      StarlarkValue starlarkFailingDownload = (StarlarkValue) failingDownload;
      failingFuture.completeExceptionally(new Throwable());

      // Check debugPrint shows FAILED.
      Printer failingPrinter = new Printer();
      starlarkFailingDownload.debugPrint(failingPrinter, starlarkThread);
      assertThat(failingPrinter.toString()).contains("(state: FAILED)");
    }
  }

  /**
   * Tests calling the async <code>repository_ctx.download()</code>, then calling <code>wait</code>
   * on the pending download and checking the return values.
   */
  @Test
  public void download_asyncWait_returnValue() throws Exception {
    // Setup a Starlark Context for testing.
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path testPath = fs.getPath("/test");
    testPath.createDirectory();
    testPath.getRelative("output").createDirectory();

    // Simulate the downloaded file.
    Path testFile = fs.getPath("/test/output/file.gz");
    OutputStream o = testFile.getOutputStream();
    o.write(getEmptyTarGzBytes());
    o.close();

    when(downloadManager.finalizeDownload(any())).thenReturn(testFile);
    when(downloadManager.startDownload(
            /* executorService= */ any(),
            /* originalUrls= */ any(),
            /* headers= */ any(),
            /* authHeaders= */ any(),
            /* checksum= */ any(),
            /* canonicalId= */ any(),
            /* type= */ any(),
            /* output= */ any(),
            /* clientEnv= */ any(),
            /* context= */ any(),
            /* downloadPhaser= */ any(),
            /* mayHardlink= */ anyBoolean()))
        .thenReturn(new CompletableFuture<>());

    try (StarlarkBaseExternalContext sbec = setupStarlarkContext(testPath)) {
      when(environment.getListener()).thenReturn(extendedEventHandler);

      Object download =
          sbec.download(
              /* url= */ "http://example.com/file",
              /* output= */ "/test/output",
              /* sha256= */ SHA256_EMPTY_GZ_FILE,
              /* executable= */ false,
              /* allowFail= */ true,
              /* canonicalId= */ "id",
              /* authUnchecked= */ Dict.<String, Dict<String, Object>>builder().buildImmutable(),
              /* headersUnchecked= */ Dict.<String, Dict<String, Object>>builder().buildImmutable(),
              /* integrity= */ "",
              /* block= */ false,
              /* thread= */ starlarkThread);
      Object returnValue = callStarlarkMethod(download, "wait");
      assertThat(returnValue).isInstanceOf(StructImpl.class);
      StructImpl struct = (StructImpl) returnValue;
      Printer p = new Printer();
      struct.repr(p, StarlarkSemantics.DEFAULT);
      assertThat(struct.getValue("success", Boolean.class)).isEqualTo(true);
      assertThat(struct.getValue("error", String.class)).isNull();
      assertThat(struct.getValue("sha256", String.class)).isEqualTo(SHA256_EMPTY_GZ_FILE);
      assertThat(struct.getValue("integrity", String.class))
          .isEqualTo(
              "sha256-"
                  + Base64.getEncoder()
                      .encodeToString(HexFormat.of().parseHex(SHA256_EMPTY_GZ_FILE)));
      assertThat(struct.getValue("size_bytes", StarlarkInt.class))
          .isEqualTo(StarlarkInt.of(emptyTarGzBytes.length));
    }
  }

  /**
   * Tests the return value of <code>repository_ctx.download_and_extract()</code> when successful.
   */
  @Test
  public void downloadAndExtract_successReturnValue() throws Exception {
    when(environment.getListener()).thenReturn(extendedEventHandler);
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path testPath = fs.getPath("/test");
    testPath.createDirectory();
    fs.getPath("/test/output").createDirectory();
    Path testFile = fs.getPath("/test/output/path.gz");

    OutputStream o = testFile.getOutputStream();
    o.write(emptyTarGzBytes);
    o.close();
    when(downloadManager.finalizeDownload(any())).thenReturn(testFile);

    when(downloadManager.startDownload(
            /* executorService= */ any(),
            /* originalUrls= */ any(),
            /* headers= */ any(),
            /* authHeaders= */ any(),
            /* checksum= */ any(),
            /* canonicalId= */ any(),
            /* type= */ any(),
            /* output= */ any(),
            /* clientEnv= */ any(),
            /* context= */ any(),
            /* downloadPhaser= */ any(),
            /* mayHardlink= */ anyBoolean()))
        .thenReturn(new CompletableFuture<>());
    try (StarlarkBaseExternalContext sbec = setupStarlarkContext(testPath)) {
      StructImpl struct =
          sbec.downloadAndExtract(
              /* url= */ "http://example.com/file.txt.gz",
              /* output= */ "/test/output",
              /* sha256= */ SHA256_EMPTY_GZ_FILE,
              /* type= */ "gz",
              /* stripPrefix= */ "",
              /* allowFail= */ false,
              /* canonicalId= */ "id",
              /* authUnchecked= */ Dict.<String, Dict<String, Object>>builder().buildImmutable(),
              /* headersUnchecked= */ Dict.<String, String>builder().buildImmutable(),
              /* integrity= */ "",
              /* renameFiles= */ Dict.<String, String>builder().buildImmutable(),
              /* oldStripPrefix= */ "",
              /* stripComponentsI= */ StarlarkInt.of(0),
              /* thread= */ starlarkThread);
      Printer p = new Printer();
      struct.repr(p, StarlarkSemantics.DEFAULT);
      assertThat(struct.getValue("success", Boolean.class)).isEqualTo(true);
      assertThat(struct.getValue("error", String.class)).isNull();
      assertThat(struct.getValue("sha256", String.class)).isEqualTo(SHA256_EMPTY_GZ_FILE);
      assertThat(struct.getValue("integrity", String.class))
          .isEqualTo(
              "sha256-"
                  + Base64.getEncoder()
                      .encodeToString(HexFormat.of().parseHex(SHA256_EMPTY_GZ_FILE)));
      assertThat(struct.getValue("size_bytes", StarlarkInt.class))
          .isEqualTo(StarlarkInt.of(emptyTarGzBytes.length));
    }
  }

  /**
   * Tests the return value of <code>download_and_extract</code> when <code>allow_fail</code> is
   * <code>true</code> and a failure occurs.
   */
  @Test
  public void downloadAndExtract_allowFail_unsuccessfulReturnValue() throws Exception {
    when(environment.getListener()).thenReturn(extendedEventHandler);
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path testPath = fs.getPath("/test");
    testPath.createDirectory();
    fs.getPath("/test/output").createDirectory();
    when(downloadManager.finalizeDownload(any())).thenThrow(new IOException("test exception"));
    when(downloadManager.startDownload(
            /* executorService= */ any(),
            /* originalUrls= */ any(),
            /* headers= */ any(),
            /* authHeaders= */ any(),
            /* checksum= */ any(),
            /* canonicalId= */ any(),
            /* type= */ any(),
            /* output= */ any(),
            /* clientEnv= */ any(),
            /* context= */ any(),
            /* downloadPhaser= */ any(),
            /* mayHardlink= */ anyBoolean()))
        .thenReturn(new CompletableFuture<>());
    try (StarlarkBaseExternalContext sbec = setupStarlarkContext(testPath)) {
      StructImpl struct =
          sbec.downloadAndExtract(
              /* url= */ "http://example.com/file.txt.gz",
              /* output= */ "/test/output",
              /* sha256= */ SHA256_EMPTY_FILE,
              /* type= */ "gz",
              /* stripPrefix= */ "",
              /* allowFail= */ true,
              /* canonicalId= */ "id",
              /* authUnchecked= */ Dict.<String, Dict<String, Object>>builder().buildImmutable(),
              /* headersUnchecked= */ Dict.<String, String>builder().buildImmutable(),
              /* integrity= */ "",
              /* renameFiles= */ Dict.<String, String>builder().buildImmutable(),
              /* oldStripPrefix= */ "",
              /* stripComponentsI= */ StarlarkInt.of(0),
              /* thread= */ starlarkThread);
      Printer p = new Printer();
      struct.repr(p, StarlarkSemantics.DEFAULT);
      assertThat(struct.getValue("success", Boolean.class)).isEqualTo(false);
      assertThat(struct.getValue("error", String.class))
          .isEqualTo("java.io.IOException: test exception");
    }
  }

  @Test
  public void docSupportedFormats() {
    String expected = DecompressorValue.readableSupportedFormats("\"", "\"", "or");
    String observed = StarlarkBaseExternalContext.SUPPORTED_DECOMPRESSION_FORMATS;
    String copyPasteCode =
        "  static final String SUPPORTED_DECOMPRESSION_FORMATS =\n"
            + "\"\"\"\n"
            + WordUtils.wrap(
                expected,
                /* wrapLength= */ 80,
                /* newLineStr= */ " \\\n",
                /* wrapLongWords= */ false)
            + "\\\n\"\"\";";

    if (!observed.equals(expected)) {
      fail(
          String.format(
              """


              Expected:
              \t%1$s
              Got:
              \t%2$s

              Copy-paste string to replace in StarlarkBaseExternalContext.java: \s

              %3$s
              """,
              expected, observed, copyPasteCode));
    }
  }
}
