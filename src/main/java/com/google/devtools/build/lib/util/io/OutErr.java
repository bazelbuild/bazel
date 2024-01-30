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

import com.google.common.base.Preconditions;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;

/**
 * A pair of output streams to be used for redirecting the output and error streams of a subprocess.
 */
public class OutErr implements Closeable {

  private final OutputStream out;
  private final OutputStream err;

  public static final OutErr SYSTEM_OUT_ERR = create(System.out, System.err);

  /** Creates a new OutErr instance from the specified output and error streams. */
  public static OutErr create(OutputStream out, OutputStream err) {
    return new OutErr(out, err);
  }

  protected OutErr(OutputStream out, OutputStream err) {
    this.out = Preconditions.checkNotNull(out);
    this.err = Preconditions.checkNotNull(err);
  }

  @Override
  public void close() throws IOException {
    // Ensure that we close both out and err even if one throws.
    try {
      out.close();
    } finally {
      if (out != err) {
        err.close();
      }
    }
  }

  /** Returns a {@link SystemPatcher} that uses this instance's out and err streams. */
  public final SystemPatcher getSystemPatcher() {
    return new SystemPatcher(out, err);
  }

  /**
   * Temporarily patches {@link System#out} and {@link System#err} with custom streams.
   *
   * <p>{@link #start} is called to signal the beginning of the scope of the patch. {@link #close}
   * ends the scope of the patch, returning {@link System#out} and {@link System#err} to what they
   * were when this instance was instantiated.
   */
  public static class SystemPatcher implements AutoCloseable {
    private final PrintStream savedOut;
    private final PrintStream savedErr;
    private final SwitchingPrintStream outPatch;
    private final SwitchingPrintStream errPatch;

    private SystemPatcher(OutputStream overrideOut, OutputStream overrideErr) {
      this.savedOut = System.out;
      this.savedErr = System.err;
      this.outPatch = new SwitchingPrintStream(overrideOut);
      this.errPatch = new SwitchingPrintStream(overrideErr);
    }

    public void start() {
      System.setOut(outPatch);
      System.setErr(errPatch);
    }

    @Override
    public void close() {
      System.setOut(savedOut);
      System.setErr(savedErr);
      outPatch.switchBackTo(savedOut);
      errPatch.switchBackTo(savedErr);
    }
  }

  /**
   * Starts by streaming to {@code override}, then switches back to {@code saved}.
   *
   * <p>The switching strategy is used to guard against memory leaks. For example, if {@code
   * override} is passed directly to {@link System#setErr}, anyone may retain a reference to it via
   * {@link System#err}. Instead, they will get a reference to this class, which frees up {@code
   * override} in {@link #switchBackTo}.
   */
  private static final class SwitchingPrintStream extends PrintStream {

    private SwitchingPrintStream(OutputStream override) {
      super(override, /*autoFlush=*/ true);
    }

    private void switchBackTo(OutputStream saved) {
      out = saved;
    }
  }

  public OutputStream getOutputStream() {
    return out;
  }

  public OutputStream getErrorStream() {
    return err;
  }

  /** Writes the specified string to the output stream, and flushes. */
  public void printOut(String s) {
    PrintWriter writer = new PrintWriter(out, true);
    writer.print(s);
    writer.flush();
  }

  public void printOutLn(String s) {
    printOut(s + "\n");
  }

  /** Writes the specified string to the error stream, and flushes. */
  public void printErr(String s) {
    PrintWriter writer = new PrintWriter(err, true);
    writer.print(s);
    writer.flush();
  }

  public void printErrLn(String s) {
    printErr(s + "\n");
  }
}
