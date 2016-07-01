// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.shell;

import com.google.devtools.build.lib.shell.SubprocessBuilder.StreamAction;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.ProcessBuilder.Redirect;

/**
 * A subprocess factory that uses {@link java.lang.ProcessBuilder}.
 */
public class JavaSubprocessFactory implements Subprocess.Factory {

  /**
   * A subprocess backed by a {@link java.lang.Process}.
   */
  private static class JavaSubprocess implements Subprocess {
    private final Process process;

    private JavaSubprocess(Process process) {
      this.process = process;
    }

    @Override
    public boolean destroy() {
      process.destroy();
      return true;
    }

    @Override
    public int exitValue() {
      return process.exitValue();
    }

    @Override
    public int waitFor() throws InterruptedException {
      return process.waitFor();
    }

    @Override
    public OutputStream getOutputStream() {
      return process.getOutputStream();
    }

    @Override
    public InputStream getErrorStream() {
      return process.getErrorStream();
    }

    @Override
    public InputStream getInputStream() {
      return process.getInputStream();
    }
  }

  public static final JavaSubprocessFactory INSTANCE = new JavaSubprocessFactory();

  private JavaSubprocessFactory() {
    // We are a singleton
  }

  @Override
  public Subprocess create(SubprocessBuilder params) throws IOException {
    ProcessBuilder builder = new ProcessBuilder();
    builder.command(params.getArgv());
    if (params.getEnv() != null) {
      builder.environment().clear();
      builder.environment().putAll(params.getEnv());
    }

    builder.redirectOutput(getRedirect(params.getStdout(), params.getStdoutFile()));
    builder.redirectError(getRedirect(params.getStderr(), params.getStderrFile()));
    builder.directory(params.getWorkingDirectory());

    return new JavaSubprocess(builder.start());
  }

  /**
   * Returns a {@link ProcessBuilder.Redirect} appropriate for the parameters. If a file redirected
   * to exists, deletes the file before redirecting to it.
   */
  private Redirect getRedirect(StreamAction action, File file) throws IOException {
    switch (action) {
      case DISCARD:
        return Redirect.to(new File("/dev/null"));

      case REDIRECT:
        // We need to use Redirect.appendTo() here, because on older Linux kernels writes are
        // otherwise not atomic and might result in lost log messages:
        // https://lkml.org/lkml/2014/3/3/308
        if (file.exists()) {
          file.delete();
        }
        return Redirect.appendTo(file);

      case STREAM:
        return Redirect.PIPE;

      default:
        throw new IllegalStateException();
    }
  }
}
