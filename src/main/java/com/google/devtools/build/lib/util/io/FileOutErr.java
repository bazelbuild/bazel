// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util.io;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.io.ByteStreams;
import com.google.common.primitives.Bytes;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * An implementation of {@link OutErr} that captures all out/err output into
 * a file for stdout and a file for stderr. The files are only created if any
 * output is made.
 * The OutErr assumes that the directory that will contain the output file
 * must exist.
 *
 * You should not use this object from multiple different threads.
 */
// Note that it should be safe to treat the Output and Error streams within a FileOutErr each as
// individually ThreadCompatible.
@ThreadSafety.ThreadCompatible
public class FileOutErr extends OutErr {

  private final AtomicInteger childCount = new AtomicInteger();

  /**
   * Create a new FileOutErr that will write its input,
   * if any, to the files specified by stdout/stderr.
   *
   * No other process may write to the files,
   *
   * @param stdout The file for the stdout of this outErr
   * @param stderr The file for the stderr of this outErr
   */
  public FileOutErr(Path stdout, Path stderr) {
    this(new FileRecordingOutputStream(stdout), new FileRecordingOutputStream(stderr));
  }

  /**
   * Creates a new FileOutErr that writes its input to the file specified by output. Both
   * stdout/stderr will be copied into the single file.
   *
   * @param output The file for the both stdout and stderr of this outErr.
   */
  public FileOutErr(Path output) {
    // We don't need to create a synchronized funnel here, like in the OutErr -- The
    // respective functions in the FileRecordingOutputStream take care of locking.
    this(new FileRecordingOutputStream(output));
  }

  protected FileOutErr(AbstractFileRecordingOutputStream out,
                       AbstractFileRecordingOutputStream err) {
    super(out, err);
  }

  /**
   * Creates a new FileOutErr that discards its input. Useful
   * for testing purposes.
   */
  @VisibleForTesting
  public FileOutErr() {
    this(new NullFileRecordingOutputStream());
  }

  private FileOutErr(OutputStream stream) {
    // We need this function to duplicate the single new object into both arguments
    // of the super-constructor.
    super(stream, stream);
  }

  /**
   * Returns true if any output was recorded.
   */
  public boolean hasRecordedOutput() {
    return getFileOutputStream().hasRecordedOutput() || getFileErrorStream().hasRecordedOutput();
  }

  /**
   * Returns true if output was recorded on stdout.
   */
  public boolean hasRecordedStdout() {
    return getFileOutputStream().hasRecordedOutput();
  }

  /**
   * Returns true if output was recorded on stderr.
   */
  public boolean hasRecordedStderr() {
    return getFileErrorStream().hasRecordedOutput();
  }

  /**
   * Returns the {@link Path} this OutErr uses to buffer stdout
   *
   * <p>The user must ensure that no other process is writing to the files at time of creation.
   *
   * @return the path object with the contents of stdout
   */
  public Path getOutputPath() {
    return getFileOutputStream().getFile();
  }

  /** Returns the length of the stdout contents. */
  public long outSize() throws IOException {
    return getFileOutputStream().getRecordedOutputSize();
  }

  /**
   * Returns the {@link Path} this OutErr uses to buffer stderr.
   *
   * @return the path object with the contents of stderr
   */
  public Path getErrorPath() {
    return getFileErrorStream().getFile();
  }

  public byte[] outAsBytes() {
    return getFileOutputStream().getRecordedOutput();
  }

  @VisibleForTesting
  public String outAsLatin1() {
    return new String(outAsBytes(), StandardCharsets.ISO_8859_1);
  }

  public byte[] errAsBytes() {
    return getFileErrorStream().getRecordedOutput();
  }

  @VisibleForTesting
  public String errAsLatin1() {
    return new String(errAsBytes(), StandardCharsets.ISO_8859_1);
  }

  /** Returns the length of the stderr contents. */
  public long errSize() throws IOException {
    return getFileErrorStream().getRecordedOutputSize();
  }

  /**
   * Closes and deletes the error stream.
   */
  public void clearErr() throws IOException {
    getFileErrorStream().clear();
  }

  /**
   * Closes and deletes the out stream.
   */
  public void clearOut() throws IOException {
    getFileOutputStream().clear();
  }


  /**
   * Writes the captured out content to the given output stream,
   * avoiding keeping the entire contents in memory.
   */
  public void dumpOutAsLatin1(OutputStream out) {
    getFileOutputStream().dumpOut(out);
  }

  /**
   * Writes the captured error content to the given error stream,
   * avoiding keeping the entire contents in memory.
   */
  public void dumpErrAsLatin1(OutputStream out) {
    getFileErrorStream().dumpOut(out);
  }

  /**
   * Writes the captured content to the given {@link FileOutErr},
   * avoiding keeping the entire contents in memory.
   */
  public static void dump(FileOutErr from, FileOutErr to) {
    from.dumpOutAsLatin1(to.getOutputStream());
    from.dumpErrAsLatin1(to.getErrorStream());
  }

  private AbstractFileRecordingOutputStream getFileOutputStream() {
    return (AbstractFileRecordingOutputStream) getOutputStream();
  }

  private AbstractFileRecordingOutputStream getFileErrorStream() {
    return (AbstractFileRecordingOutputStream) getErrorStream();
  }

  @ThreadSafe
  public FileOutErr childOutErr() {
    int index = childCount.getAndIncrement();
    Path outPath = getFileOutputStream().getFileUnsafe();
    Path errPath = getFileErrorStream().getFileUnsafe();
    if (outPath == null || errPath == null) {
      return new FileOutErr();
    }
    return new FileOutErr(
        outPath.getParentDirectory().getRelative(outPath.getBaseName() + "-" + index),
        errPath.getParentDirectory().getRelative(errPath.getBaseName() + "-" + index));
  }
  /**
   * An abstract supertype for the two other inner classes in this type
   * to implement streams that can write to a file.
   */
  private abstract static class AbstractFileRecordingOutputStream extends OutputStream {

    /**
     * Returns true if this FileRecordingOutputStream has encountered an error.
     *
     * @return true there was an error, false otherwise.
     */
    abstract boolean hadError();

    /**
     * Returns the file this FileRecordingOutputStream is writing to.
     */
    abstract Path getFile();

    /**
     * Returns true if the FileOutErr has stored output.
     */
    abstract boolean hasRecordedOutput();

    /** Returns the output this AbstractFileOutErr has recorded. */
    abstract byte[] getRecordedOutput();

    /** Returns the size of the recorded output. */
    abstract long getRecordedOutputSize() throws IOException;

    /**
     * Writes the output to the given output stream,
     * avoiding keeping the entire contents in memory.
     */
    abstract void dumpOut(OutputStream out);

    abstract Path getFileUnsafe();

    abstract boolean mightHaveOutput();

    /** Closes and deletes the output. */
    abstract void clear() throws IOException;
  }

  /**
   * An output stream that pretends to capture all its output into a file,
   * but instead discards it.
   */
  private static class NullFileRecordingOutputStream extends AbstractFileRecordingOutputStream {

    NullFileRecordingOutputStream() {
    }

    @Override
    boolean hadError() {
      return false;
    }

    @Override
    Path getFile() {
      return null;
    }

    @Override
    Path getFileUnsafe() {
      return null;
    }

    @Override
    boolean hasRecordedOutput() {
      return false;
    }

    @Override
    byte[] getRecordedOutput() {
      return new byte[] {};
    }

    @Override
    long getRecordedOutputSize() {
      return 0;
    }

    @Override
    boolean mightHaveOutput() {
      return false;
    }

    @Override
    void dumpOut(OutputStream out) {
      return;
    }

    @Override
    public void clear() {
    }

    @Override
    public void write(byte[] b, int off, int len) {
    }

    @Override
    public void write(int b) {
    }

    @Override
    public void write(byte[] b) {
    }
  }

  /**
   * An output stream that captures all output into a file. The file is created only if output is
   * received.
   *
   * The user must take care that nobody else is writing to the file that is backing the output
   * stream.
   *
   * The write() methods of type are synchronized to ensure that writes from different threads are
   * not mixed up. Note that this class is otherwise
   * {@link com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible}. Only the
   * write() methods are allowed to be concurrently, and only concurrently with each other. All
   * other calls must be serialized.
   *
   * The outputStream is here only for the benefit of the pumping IO we're currently using for
   * execution - Once that is gone we can remove this output stream and fold its code into the
   * FileOutErr.
   */
  @ThreadSafety.ThreadCompatible
  protected static class FileRecordingOutputStream extends AbstractFileRecordingOutputStream {

    private final Path outputFile;
    private OutputStream outputStream;
    private String error;
    private boolean mightHaveOutput = false;

    protected FileRecordingOutputStream(Path outputFile) {
      this.outputFile = outputFile;
    }

    @Override
    boolean hadError() {
      return error != null;
    }

    @Override
    Path getFile() {
      // The caller is getting a reference to the filesystem path, so conservatively assume the
      // file has been modified.
      markDirty();
      return outputFile;
    }

    @Override
    Path getFileUnsafe() {
      return outputFile;
    }

    private void markDirty() {
      mightHaveOutput = true;
    }

    private OutputStream getOutputStream() throws IOException {
      // you should hold the lock before you invoke this method
      if (outputStream == null) {
        outputStream = outputFile.getOutputStream();
      }
      return outputStream;
    }

    private boolean hasOutputStream() {
      return outputStream != null;
    }

    @Override
    public synchronized void clear() throws IOException {
      close();
      outputStream = null;
      outputFile.delete();
      mightHaveOutput = false;
    }

    /**
     * Called whenever the FileRecordingOutputStream finds an error.
     */
    protected void recordError(IOException exception) {
      String newErrorText = exception.getMessage();
      error = (error == null) ? newErrorText : error + "\n" + newErrorText;
    }

    @Override
    boolean hasRecordedOutput() {
      try {
        return getRecordedOutputSize() > 0;
      } catch (IOException ex) {
        recordError(ex);
        return true;
      }
    }

    @Override
    boolean mightHaveOutput() {
      return mightHaveOutput;
    }

    @Override
    byte[] getRecordedOutput() {
      byte[] bytes = null;
      try {
        if (mightHaveOutput && getFile().exists()) {
          bytes = FileSystemUtils.readContent(getFile());
        }
      } catch (IOException ex) {
        recordError(ex);
      }

      if (hadError()) {
        byte[] errorBytes = error.getBytes(StandardCharsets.ISO_8859_1);
        if (bytes == null) {
          bytes = errorBytes;
        } else {
          bytes = Bytes.concat(bytes, errorBytes);
        }
      }
      return bytes == null ? new byte[] {} : bytes;
    }

    @Override
    long getRecordedOutputSize() throws IOException {
      if (hadError()) {
        return error.length();
      }
      if (!mightHaveOutput) {
        return 0;
      }
      try {
        return outputFile.getFileSize();
      } catch (FileNotFoundException e) {
        return 0;
      } catch (IOException e) {
        recordError(e);
        throw e;
      }
    }

    @Override
    void dumpOut(OutputStream out) {
      try {
        if (mightHaveOutput && getFile().exists()) {
          try (InputStream in = getFile().getInputStream()) {
            ByteStreams.copy(in, out);
            out.flush();
          }
        }
      } catch (IOException ex) {
        recordError(ex);
      }

      if (hadError()) {
        PrintStream ps = new PrintStream(out);
        ps.print(error);
        ps.flush();
      }
    }

    @Override
    public synchronized void write(byte[] b, int off, int len) {
      if (len > 0) {
        markDirty();
        try {
          getOutputStream().write(b, off, len);
        } catch (IOException ex) {
          recordError(ex);
        }
      }
    }

    @Override
    public synchronized void write(int b) {
      markDirty();
      try {
        getOutputStream().write(b);
      } catch (IOException ex) {
        recordError(ex);
      }
    }

    @Override
    public synchronized void write(byte[] b) throws IOException {
      if (b.length > 0) {
        markDirty();
        getOutputStream().write(b);
      }
    }

    @Override
    public synchronized void flush() throws IOException {
      if (hasOutputStream()) {
        getOutputStream().flush();
      }
    }

    @Override
    public synchronized void close() throws IOException {
      if (hasOutputStream()) {
        getOutputStream().close();
      }
    }
  }
}
