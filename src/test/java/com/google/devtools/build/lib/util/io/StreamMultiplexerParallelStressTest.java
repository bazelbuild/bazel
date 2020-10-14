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

import com.google.common.io.ByteStreams;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Exercise {@link StreamMultiplexer} in a parallel setting and ensure there's
 * no corruption.
 */
@RunWith(JUnit4.class)
public class StreamMultiplexerParallelStressTest {

  /**
   * Characters that could likely cause corruption (they're used as control
   * characters).
   */
  char[] toughCharsToTry = {'\n', '@', '1', '2', '\0', '0'};

  /**
   * We use a demultiplexer as a simple checker only - that is, we don't care what the demultiplexer
   * writes, but we are taking advantage of its built in error checking.
   */
  OutputStream devNull = ByteStreams.nullOutputStream();

  StreamDemultiplexer demux = new StreamDemultiplexer((byte) 1, devNull, devNull, devNull);

  /**
   * The multiplexer under test.
   */
  StreamMultiplexer mux = new StreamMultiplexer(demux);

  /**
   * Streams is the out / err / control output streams of the multiplexer which
   * we will write to in parallel.
   */
  OutputStream[] streams = {
      mux.createStdout(), mux.createStderr(), mux.createControl()};

  /**
   * We will create a bunch of threads that write random data to the streams of
   * the mux.
   */
  class RandomDataPump implements Callable<Object> {

    private Random random;

    public RandomDataPump(int threadId) {
      random = new Random(threadId * 0xdeadbeefL);
    }

    @Override
    public Object call() throws Exception {
      Thread.yield();
      OutputStream out = streams[random.nextInt(2)];
      for (int i = 0; i < 10000; i++) {
          switch (random.nextInt(5)) {
          case 0:
            out.write(random.nextInt());
            break;
          case 1:
            int index = random.nextInt(toughCharsToTry.length);
            out.write(toughCharsToTry[index]);
            break;
          case 2:
            byte[] buffer = new byte[random.nextInt(312)];
            random.nextBytes(buffer);
            out.write(buffer);
            break;
          case 3:
            out.flush();
            break;
          case 4:
            out = streams[random.nextInt(3)];
            break;
          }
      }
      return null;
    }
  }

  @Test
  public void testSingleThreadedStress() throws Exception {
    new RandomDataPump(1).call();
  }

  @Test
  public void testMultiThreadedStress()
      throws InterruptedException, ExecutionException {
    ExecutorService service = Executors.newFixedThreadPool(50);

    List<Future<?>> futures = new ArrayList<>();
    for (int threadId = 0; threadId < 50; threadId++) {
      futures.add(service.submit(new RandomDataPump(threadId)));
    }
    for (Future<?> future : futures) {
      future.get();
    }
  }

}
