// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import java.io.IOException;
import javax.annotation.Nullable;

/** A callback that can receive a finish event when there are no more partial results */
@ThreadCompatible
public abstract class OutputFormatterCallback<T> implements Callback<T> {

  private IOException ioException;

  /**
   * This method will be called before any partial result are available.
   *
   * <p>It should be used for opening resources or sending a header to the output.
   */
  public void start() throws IOException {
  }

  /**
   * Same as start but for closing resources or writing a footer.
   *
   * <p>Will be called even in the case of an error.
   */
  public void close(boolean failFast) throws InterruptedException, IOException {
  }

  /**
   * Note that {@link Callback} interface does not throw IOExceptions. What this implementation does
   * instead is throw {@code InterruptedException} and store the {@code IOException} in the {@code
   * ioException} field. Users of this class should check on InterruptedException the field to
   * disambiguate between real interruptions or IO Exceptions.
   */
  @Override
  public void process(Iterable<T> partialResult) throws QueryException, InterruptedException {
    try {
      processOutput(partialResult);
    } catch (IOException e) {
      ioException = e;
      throw new InterruptedException("Interrupting due to a IOException in the OutputFormatter");
    }
  }

  public abstract void processOutput(Iterable<T> partialResult)
      throws IOException, InterruptedException;

  @Nullable
  public IOException getIoException() {
    return ioException;
  }

  /**
   * Use an {@code OutputFormatterCallback} with an already computed set of targets. Note that this
   * does not work in stream mode, as the {@code targets} would already be computed.
   *
   * <p>The intended usage of this method is to use {@code StreamedFormatter} formaters in non
   * streaming contexts.
   */
  public static <T> void processAllTargets(OutputFormatterCallback<T> callback,
      Iterable<T> targets) throws IOException, InterruptedException {
    boolean failFast = true;
    try {
      callback.start();
      callback.process(targets);
      failFast = false;
    } catch (InterruptedException e) {
      IOException ioException = callback.getIoException();
      if (ioException != null) {
        throw ioException;
      }
      throw e;
    } catch (QueryException e) {
      throw new IllegalStateException("This should not happen, as we are not running any query,"
          + " only printing the results:" + e.getMessage(), e);
    } finally {
      callback.close(failFast);
    }
  }
}
