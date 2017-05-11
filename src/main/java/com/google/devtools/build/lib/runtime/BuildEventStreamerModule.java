// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.buildeventstream.transports.BuildEventTransportFactory.createFromOptions;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;


/** Module responsible for configuring BuildEventStreamer and transports. */
public class BuildEventStreamerModule extends BlazeModule {

  private CommandEnvironment commandEnvironment;

  private static class BuildEventRecorder {
    private final List<BuildEvent> events = new ArrayList<>();

    @Subscribe
    public void buildEvent(BuildEvent event) {
      events.add(event);
    }

    List<BuildEvent> getEvents() {
      return events;
    }
  }

  private BuildEventRecorder buildEventRecorder;

  /**
   * {@link OutputStream} suitably synchonized for producer-consumer use cases.
   * The method {@link #readAndReset()} allows to read the bytes accumulated so far
   * and simultaneously truncate precisely the bytes read. Moreover, upon such a reset
   * the amount of memory retained is reset to a small constant. This is a difference
   * with resecpt to the behaviour of the standard classes {@link ByteArrayOutputStream}
   * which only resets the index but keeps the array. This difference matters, as we need
   * to support output peeks without retaining this ammount of memory for the rest of the
   * build.
   */
  private static class SynchronizedOutputStream extends OutputStream {

    // The maximal amount of bytes we intend to store in the buffer. However,
    // the requirement that a single write be written in one go is more important,
    // so the actual size we store in this buffer can be the maximum (not the sum)
    // of this value and the amount of bytes written in a single call to the
    // {@link write(byte[] buffer, int offset, int count)} method.
    private static final long MAX_BUFFERED_LENGTH = 10 * 1024;

    private byte[] buf;
    private long count;
    private boolean discardAll;

    // The event streamer that is supposed to flush stdout/stderr.
    private BuildEventStreamer streamer;

    SynchronizedOutputStream() {
      buf = new byte[64];
      count = 0;
      discardAll = false;
    }

    void registerStreamer(BuildEventStreamer streamer) {
      this.streamer = streamer;
    }

    public synchronized void setDiscardAll() {
      discardAll = true;
      count = 0;
      buf = null;
    }

    /**
     * Read the contents of the stream and simultaneously clear them. Also, reset the amount of
     * memory retained to a constant amount.
     */
    synchronized String readAndReset() {
      String content = new String(buf, 0, (int) count, UTF_8);
      buf = new byte[64];
      count = 0;
      return content;
    }

    @Override
    public void write(int oneByte) throws IOException {
      if (discardAll) {
        return;
      }
      // We change the dependency with respect to that of the super class: write(int)
      // now calls write(int[], int, int) which is implemented without any dependencies.
      write(new byte[] {(byte) oneByte}, 0, 1);
    }

    @Override
    public void write(byte[] buffer, int offset, int count) throws IOException {
      // As we base the less common write(int) on this method, we may not depend not call write(int)
      // directly or indirectly (e.g., by calling super.write(int[], int, int)).
      synchronized (this) {
        if (discardAll) {
          return;
        }
      }
      boolean shouldFlush = false;
      // As we have to do the flushing outside the synchronized block, we have to expect
      // other writes to come immediately after flushing, so we have to do the check inside
      // a while loop.
      boolean didWrite = false;
      while (!didWrite) {
        synchronized (this) {
          if (this.count + (long) count < MAX_BUFFERED_LENGTH || this.count == 0) {
            if (this.count + (long) count >= (long) buf.length) {
              // We need to increase the buffer; if within the permissible range range for array
              // sizes, we at least double it, otherwise we only increase as far as needed.
              long newsize;
              if (2 * (long) buf.length + count < (long) Integer.MAX_VALUE) {
                newsize = 2 * (long) buf.length + count;
              } else {
                newsize = this.count + count;
              }
              byte[] newbuf = new byte[(int) newsize];
              System.arraycopy(buf, 0, newbuf, 0, (int) this.count);
              this.buf = newbuf;
            }
            System.arraycopy(buffer, offset, buf, (int) this.count, count);
            this.count += (long) count;
            didWrite = true;
          } else {
            shouldFlush = true;
          }
          if (this.count >= MAX_BUFFERED_LENGTH) {
            shouldFlush = true;
          }
        }
        if (shouldFlush && streamer != null) {
          streamer.flush();
          shouldFlush = false;
        }
      }
    }
  }

  private SynchronizedOutputStream out;
  private SynchronizedOutputStream err;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.<Class<? extends OptionsBase>>of(BuildEventStreamOptions.class);
  }

  @Override
  public void checkEnvironment(CommandEnvironment commandEnvironment) {
    this.commandEnvironment = commandEnvironment;
    this.buildEventRecorder = new BuildEventRecorder();
    commandEnvironment.getEventBus().register(buildEventRecorder);
  }

  @Override
  public OutErr getOutputListener() {
    this.out = new SynchronizedOutputStream();
    this.err = new SynchronizedOutputStream();
    return OutErr.create(this.out, this.err);
  }

  @Override
  public void handleOptions(OptionsProvider optionsProvider) {
    checkState(commandEnvironment != null, "Methods called out of order");
    Optional<BuildEventStreamer> maybeStreamer =
        tryCreateStreamer(optionsProvider, commandEnvironment.getBlazeModuleEnvironment());
    if (maybeStreamer.isPresent()) {
      BuildEventStreamer streamer = maybeStreamer.get();
      commandEnvironment.getReporter().addHandler(streamer);
      commandEnvironment.getEventBus().register(streamer);
      for (BuildEvent event : buildEventRecorder.getEvents()) {
        streamer.buildEvent(event);
      }
      final SynchronizedOutputStream theOut = this.out;
      final SynchronizedOutputStream theErr = this.err;
      // out and err should be non-null at this point, as getOutputListener is supposed to
      // be always called before handleOptions. But let's still prefer a stream with no
      // stdout/stderr over an aborted build.
      streamer.registerOutErrProvider(
          new BuildEventStreamer.OutErrProvider() {
            @Override
            public String getOut() {
              if (theOut == null) {
                return null;
              }
              return theOut.readAndReset();
            }

            @Override
            public String getErr() {
              if (theErr == null) {
                return null;
              }
              return theErr.readAndReset();
            }
          });
      if (theErr != null) {
        theErr.registerStreamer(streamer);
      }
      if (theOut != null) {
        theOut.registerStreamer(streamer);
      }
    } else {
      // If there is no streamer to consume the output, we should not try to accumulate it.
      this.out.setDiscardAll();
      this.err.setDiscardAll();
    }
    commandEnvironment.getEventBus().unregister(buildEventRecorder);
    this.buildEventRecorder = null;
    this.out = null;
    this.err = null;
  }

  @VisibleForTesting
  Optional<BuildEventStreamer> tryCreateStreamer(
      OptionsProvider optionsProvider, ModuleEnvironment moduleEnvironment) {
    try {
      PathConverter pathConverter;
      if (commandEnvironment == null) {
        pathConverter = new PathConverter() {
            @Override
            public String apply(Path path) {
              return path.getPathString();
            }
          };
      } else {
        pathConverter = commandEnvironment.getRuntime().getPathToUriConverter();
      }
      BuildEventStreamOptions besOptions =
          checkNotNull(
              optionsProvider.getOptions(BuildEventStreamOptions.class),
              "Could not get BuildEventStreamOptions");
      ImmutableSet<BuildEventTransport> buildEventTransports
          = createFromOptions(besOptions, pathConverter);
      if (!buildEventTransports.isEmpty()) {
        BuildEventStreamer streamer = new BuildEventStreamer(buildEventTransports,
            commandEnvironment != null ? commandEnvironment.getReporter() : null);
        return Optional.of(streamer);
      }
    } catch (IOException e) {
      moduleEnvironment.exit(new AbruptExitException(ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e));
    }
    return Optional.absent();
  }
}
