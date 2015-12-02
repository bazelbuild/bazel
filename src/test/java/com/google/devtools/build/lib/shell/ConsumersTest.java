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
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.shell.Consumers.OutErrConsumers;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

@RunWith(JUnit4.class)
public class ConsumersTest {

  @Before
  public final void configureLogger() throws Exception  {
    // enable all log statements to ensure there are no problems with
    // logging code
    Logger.getLogger("com.google.devtools.build.lib.shell.Command").setLevel(Level.FINEST);
  }

  private static final String SECRET_MESSAGE = "This is a secret message.";

  /**
   * Tests that if an IOException occurs in an output stream, the exception
   * will be recorded and thrown when we call waitForCompletion.
   */
  @Test
  public void testAsynchronousIOExceptionInConsumerOutputStream() {

    OutputStream out = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw new IOException(SECRET_MESSAGE);
      }
    };
    OutErrConsumers outErr = Consumers.createStreamingConsumers(out, out);
    ByteArrayInputStream outInput = new ByteArrayInputStream(new byte[]{'a'});
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    try {
      outErr.waitForCompletion();
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessage(SECRET_MESSAGE);
    }
  }

  /**
   * Tests that if an OutOfMemeoryError occurs in an output stream, it
   * will be recorded and thrown when we call waitForCompletion.
   */
  @Test
  public void testAsynchronousOutOfMemoryErrorInConsumerOutputStream() {
    final OutOfMemoryError error = new OutOfMemoryError(SECRET_MESSAGE);
    OutputStream out = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw error;
      }
    };
    OutErrConsumers outErr = Consumers.createStreamingConsumers(out, out);
    ByteArrayInputStream outInput = new ByteArrayInputStream(new byte[]{'a'});
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    try {
      outErr.waitForCompletion();
      fail();
    } catch (IOException e) {
      fail();
    } catch (OutOfMemoryError e) {
      assertSame("OutOfMemoryError is not masked", error, e);
    }
  }

  /**
   * Tests that if an Error occurs in an output stream, the error
   * will be recorded and thrown when we call waitForCompletion.
   */
  @Test
  public void testAsynchronousErrorInConsumerOutputStream() {
    OutputStream out = new OutputStream() {
      @Override
      public void write(int b) throws IOException {
        throw new OutOfMemoryError(SECRET_MESSAGE);
      }
    };
    OutErrConsumers outErr = Consumers.createStreamingConsumers(out, out);
    ByteArrayInputStream outInput = new ByteArrayInputStream(new byte[]{'a'});
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    try {
      outErr.waitForCompletion();
      fail();
    } catch (IOException e) {
      fail();
    } catch (Error e) {
      assertThat(e).hasMessage(SECRET_MESSAGE);
    }
  }

  /**
   * Tests that if an RuntimeException occurs in an output stream, the exception
   * will be recorded and thrown when we call waitForCompletion.
   */
  @Test
  public void testAsynchronousRuntimeExceptionInConsumerOutputStream()
  throws Exception {
    OutputStream out = new OutputStream() {
      @Override
      public void write(int b) {
        throw new RuntimeException(SECRET_MESSAGE);
      }
    };
    OutErrConsumers outErr = Consumers.createStreamingConsumers(out, out);
    ByteArrayInputStream outInput = new ByteArrayInputStream(new byte[]{'a'});
    ByteArrayInputStream errInput = new ByteArrayInputStream(new byte[0]);
    outErr.registerInputs(outInput, errInput, false);
    try {
      outErr.waitForCompletion();
      fail();
    } catch (RuntimeException e) {
      assertThat(e).hasMessage(SECRET_MESSAGE);
    }
  }

  /**
   * Basically tests that Consumers#silentClose(InputStream) works properly.
   */
  @Test
  public void testSilentlyDropIOExceptionWhenClosingInputStream() {
    InputStream in = new ByteArrayInputStream(new byte[]{'a'}){
      @Override
      public void close() throws IOException {
        throw new IOException("Please ignore me!");
      }
    };
    OutErrConsumers outErr = Consumers.createDiscardingConsumers();
    outErr.registerInputs(in, in, false);
    try {
      outErr.waitForCompletion();
      // yeah!
    } catch (IOException e) {
      fail(); // this should not throw an exception, since we're silently
              // closing the output!
    }
  }

}
