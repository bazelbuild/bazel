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
package com.google.devtools.build.lib.shell;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.shell.Consumers.OutErrConsumers;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that InterruptedExceptions can't derail FutureConsumption
 * instances; well, FutureConsumption is really an implementation detail,
 * but we want to exercise this code, so what ...
 */
@RunWith(JUnit4.class)
public class FutureConsumptionTest {

  @Before
  public final void configureLogger() throws Exception  {
    // enable all log statements to ensure there are no problems with
    // logging code
    Logger.getLogger("com.google.devtools.build.lib.shell.Command").setLevel(Level.FINEST);
  }

  private OutputStream DEV_NULL = new OutputStream() {
    @Override
    public void write(int b) {}
  };

  @Test
  public void testFutureConsumptionIgnoresInterruptedExceptions()
  throws Exception {
    // Set this up so that the consumer actually have to stream stuff into
    // DEV_NULL, which the discards everything.
    OutErrConsumers outErr = Consumers.createStreamingConsumers(DEV_NULL,
                                                                DEV_NULL);

    final AtomicBoolean inputFinished = new AtomicBoolean(false);

    // We keep producing input until the other thread (the main test thread)
    // tells us to shut up ...
    InputStream outInput = new InputStream() {
      @Override
      public int read() {
        if (inputFinished.get()){
          return -1;
        }
        return 0;
      }
    };
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    // OK, this is the main test thread, which we need to interrupt *while*
    // it's waiting in outErr.waitForCompletion()
    final Thread testThread = Thread.currentThread();

    // go into a different thread, wait a bit, interrupt the test thread,
    // wait a bit, and tell the input stream to finish.
    new Thread() {
      @Override
      public void run() {
        try {
          Thread.sleep(1000);
        } catch (InterruptedException e) {}
        testThread.interrupt(); // this is what we're testing; basic
        try {
          Thread.sleep(1000);
        } catch (InterruptedException e) {}
        inputFinished.set(true);
      }
    }.start();

    outErr.waitForCompletion();
    // In addition to asserting that we were interrupted, this clears the interrupt bit of the
    // current thread, since Junit doesn't do it for us. This avoids the next test to run starting
    // in an interrupted state.
    assertThat(Thread.interrupted()).isTrue();
  }
}
