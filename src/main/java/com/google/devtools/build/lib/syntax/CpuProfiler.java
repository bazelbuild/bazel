// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.GZIPOutputStream;
import javax.annotation.Nullable;

// Overview
//
// A CPU profiler measures CPU cycles consumed by each thread.
// It does not account for time a thread is blocked in I/O
// (e.g. within a call to glob), or runnable but not actually
// running, as happens when there are more runnable threads than cores.
//
// CPU profiling requires operating system support.
// On POSIX systems, the setitimer system call causes
// the kernel to signal an application periodically.
// With the ITIMER_PROF option, setitimer delivers a
// SIGPROF signal to a running thread each time its CPU usage
// exceeds the specific quantum. A profiler builds a histogram
// of these these signals, grouped by the current program
// counter location, or more usefully by the complete stack of
// program counter locations.
//
// This profiler calls a C++ function to install a SIGPROF handler.
// Like all handlers for asynchronous signals (that is, signals not
// caused by the execution of program instructions), it is extremely
// constrained in what it may do. It cannot acquire locks, allocate
// memory, or interact with the JVM in any way. Our signal handler
// simply sends a message into a global pipe; the message records
// the operating system's identifier (tid) for the signalled thread.
//
// Reading from the other end of the pipe is a Java thread, the router.
// Its job is to map each OS tid to a StarlarkThread, if the
// thread is currently executing Starlark code, and increment
// a volatile counter in that StarlarkThread. If the thread is
// not executing Starlark code, the router discards the event.
// When a Starlark thread enters or leaves a function during profiling,
// it updates the StarlarkThread-to-OS-thread mapping consulted by the
// router.
//
// If the router does not drain the pipe in a timely manner (on the
// order of 10s; see signal handler), the signal handler prints a
// warning and discards the event.
//
// The router may induce a delay between the kernel signal and the
// thread's stack sampling, during which Starlark execution may have
// moved on to another function. Assuming uniform delay, this is
// equivalent to shifting the phase but not the frequency of CPU ticks.
// Nonetheless it may bias the profile because, for example,
// it would cause a Starlark 'sleep' function to accrue a nonzero
// number of CPU ticks that properly belong to the preceding computation.
//
// When a Starlark thread leaves any function, it reads and clears
// its counter of CPU ticks. If the counter was nonzero, the thread
// writes a copy of its stack to the profiler log in pprof form,
// which is a gzip-compressed stream of protocol messages.
//
// The profiler is inherently global to the process,
// and records the effects of all Starlark threads.
// It may be started and stopped concurrent with Starlark execution,
// allowing profiling of a portion of a long-running computation.

/** A CPU profiler for Starlark (POSIX only for now). */
final class CpuProfiler {

  static {
    JNI.load();
  }

  private final PprofWriter pprof;

  private CpuProfiler(OutputStream out, Duration period) {
    this.pprof = new PprofWriter(out, period);
  }

  // The active profiler, if any.
  @Nullable private static volatile CpuProfiler instance;

  /** Returns the active profiler, or null if inactive. */
  @Nullable
  static CpuProfiler get() {
    return instance;
  }

  // Maps OS thread ID to StarlarkThread.
  // The StarlarkThread is needed only for its cpuTicks field.
  private static final Map<Integer, StarlarkThread> threads = new ConcurrentHashMap<>();

  /**
   * Associates the specified StarlarkThread with the current OS thread. Returns the StarlarkThread
   * previously associated with it, if any.
   */
  @Nullable
  static StarlarkThread setStarlarkThread(StarlarkThread thread) {
    if (thread == null) {
      return threads.remove(gettid());
    } else {
      return threads.put(gettid(), thread);
    }
  }

  /** Start the profiler. */
  static void start(OutputStream out, Duration period) {
    if (!supported()) {
      throw new UnsupportedOperationException("this platform does not support Starlark profiling");
    }
    if (instance != null) {
      throw new IllegalStateException("profiler started twice without intervening stop");
    }

    startRouter();
    if (!startTimer(period.toNanos() / 1000L)) {
      throw new IllegalStateException("profile signal handler already in use");
    }

    instance = new CpuProfiler(out, period);
  }

  /** Stop the profiler and wait for the log to be written. */
  static void stop() throws IOException {
    if (instance == null) {
      throw new IllegalStateException("stop without start");
    }

    CpuProfiler profiler = instance;
    instance = null;

    stopTimer();

    // Finish writing the file and fail if there were any I/O errors.
    profiler.pprof.writeEnd();
  }

  /** Records a profile event. */
  void addEvent(int ticks, ImmutableList<Debug.Frame> stack) {
    pprof.writeEvent(ticks, stack);
  }

  // ---- signal router ----

  private static FileInputStream pipe;

  // Starts the routing thread if not already started (idempotent).
  // On return, it is safe to install the signal handler.
  private static synchronized void startRouter() {
    if (pipe == null) {
      pipe = new FileInputStream(createPipe());
      Thread router = new Thread(CpuProfiler::router, "SIGPROF router");
      router.setDaemon(true);
      router.start();
    }
  }

  // The Router thread routes SIGPROF events (from the pipe)
  // to the relevant StarlarkThread. Once started, it runs forever.
  //
  // TODO(adonovan): opt: a more efficient implementation of routing would be
  // to use, instead of a pipe from the signal handler to the routing thread,
  // a mapping, maintained in C++, from OS thread ID to cpuTicks pointer.
  // The {add,remove}Thread operations would update this mapping,
  // and the signal handler would read it. The mapping would have to
  // be a lock-free hash table so that it can be safely read in an
  // async signal handler. The pointer would point to the sole element
  // of direct memory buffer belonging to the StarlarkThread, allocated
  // by JNI NewDirectByteBuffer.
  // In this way, the signal handler could update the StarlarkThread directly,
  // saving 100 write+read calls per second per core.
  //
  private static void router() {
    byte[] buf = new byte[4];
    while (true) {
      try {
        int n = pipe.read(buf);
        if (n < 0) {
          throw new IllegalStateException("pipe closed");
        }
        if (n != 4) {
          throw new IllegalStateException("short read");
        }
      } catch (IOException ex) {
        throw new IllegalStateException("unexpected I/O error", ex);
      }

      int tid = int32be(buf);

      // Record a CPU tick against tid.
      //
      // It's not safe to grab the thread's stack here because the thread
      // may be changing it, so we increment the thread's counter.
      // When the thread later observes the counter is non-zero,
      // it gives us the stack by calling addEvent.
      StarlarkThread thread = threads.get(tid);
      if (thread != null) {
        thread.cpuTicks.getAndIncrement();
      }
    }
  }

  // Decodes a signed 32-bit big-endian integer from b[0:4].
  private static int int32be(byte[] b) {
    return b[0] << 24 | (b[1] & 0xff) << 16 | (b[2] & 0xff) << 8 | (b[3] & 0xff);
  }

  // --- native code (see cpu_profiler) ---

  // Reports whether the profiler is supported on this platform.
  private static native boolean supported();

  // Returns the read end of a pipe from which profile events may be read.
  // Each event is an operating system thread ID encoded as uint32be.
  private static native FileDescriptor createPipe();

  // Starts the operating system's interval timer.
  // The period must be a positive number of microseconds.
  // Returns false if SIGPROF is already in use.
  private static native boolean startTimer(long periodMicros);

  // Stops the operating system's interval timer.
  private static native void stopTimer();

  // Returns the operating system's identifier for the calling thread.
  private static native int gettid();

  // Encoder for pprof format profiles.
  // See https://github.com/google/pprof/tree/master/proto
  // We encode the protocol messages by hand to avoid
  // adding a dependency on the protocol compiler.
  private static final class PprofWriter {

    private final Duration period;
    private final long startNano;
    private GZIPOutputStream gz;
    private CodedOutputStream enc;
    private IOException error; // the first write error, if any; reported during stop()

    PprofWriter(OutputStream out, Duration period) {
      this.period = period;
      this.startNano = System.nanoTime();

      try {
        this.gz = new GZIPOutputStream(out);
        this.enc = CodedOutputStream.newInstance(gz);
        getStringID(""); // entry 0 is always ""

        // dimension and unit
        ByteArrayOutputStream unit = new ByteArrayOutputStream();
        CodedOutputStream unitEnc = CodedOutputStream.newInstance(unit);
        unitEnc.writeInt64(VALUETYPE_TYPE, getStringID("CPU"));
        unitEnc.writeInt64(VALUETYPE_UNIT, getStringID("microseconds"));
        unitEnc.flush();

        // informational fields of Profile
        enc.writeByteArray(PROFILE_SAMPLE_TYPE, unit.toByteArray());
        // magnitude of sampling period:
        enc.writeInt64(PROFILE_PERIOD, period.toNanos() / 1000L);
        // dimension and unit of period:
        enc.writeByteArray(PROFILE_PERIOD_TYPE, unit.toByteArray());
        // start (real) time of profile:
        enc.writeInt64(PROFILE_TIME_NANOS, System.currentTimeMillis() * 1000000L);
      } catch (IOException ex) {
        this.error = ex;
      }
    }

    synchronized void writeEvent(int ticks, ImmutableList<Debug.Frame> stack) {
      if (this.error == null) {
        try {
          ByteArrayOutputStream sample = new ByteArrayOutputStream();
          CodedOutputStream sampleEnc = CodedOutputStream.newInstance(sample);
          sampleEnc.writeInt64(SAMPLE_VALUE, ticks * period.toNanos() / 1000L);
          for (Debug.Frame fr : stack.reverse()) {
            sampleEnc.writeUInt64(SAMPLE_LOCATION_ID, getLocationID(fr));
          }
          sampleEnc.flush();
          enc.writeByteArray(PROFILE_SAMPLE, sample.toByteArray());
        } catch (IOException ex) {
          this.error = ex;
        }
      }
    }

    synchronized void writeEnd() throws IOException {
      long endNano = System.nanoTime();
      try {
        enc.writeInt64(PROFILE_DURATION_NANOS, endNano - startNano);
        enc.flush();
        if (this.error != null) {
          throw this.error; // retained from an earlier error
        }
      } finally {
        gz.close();
      }
    }

    // Field numbers from pprof protocol.
    // See https://github.com/google/pprof/blob/master/proto/profile.proto
    private static final int PROFILE_SAMPLE_TYPE = 1; // repeated ValueType
    private static final int PROFILE_SAMPLE = 2; // repeated Sample
    private static final int PROFILE_MAPPING = 3; // repeated Mapping
    private static final int PROFILE_LOCATION = 4; // repeated Location
    private static final int PROFILE_FUNCTION = 5; // repeated Function
    private static final int PROFILE_STRING_TABLE = 6; // repeated string
    private static final int PROFILE_TIME_NANOS = 9; // int64
    private static final int PROFILE_DURATION_NANOS = 10; // int64
    private static final int PROFILE_PERIOD_TYPE = 11; // ValueType
    private static final int PROFILE_PERIOD = 12; // int64
    private static final int VALUETYPE_TYPE = 1; // int64
    private static final int VALUETYPE_UNIT = 2; // int64
    private static final int SAMPLE_LOCATION_ID = 1; // repeated uint64
    private static final int SAMPLE_VALUE = 2; // repeated int64
    private static final int SAMPLE_LABEL = 3; // repeated Label
    private static final int LABEL_KEY = 1; // int64
    private static final int LABEL_STR = 2; // int64
    private static final int LABEL_NUM = 3; // int64
    private static final int LABEL_NUM_UNIT = 4; // int64
    private static final int LOCATION_ID = 1; // uint64
    private static final int LOCATION_MAPPING_ID = 2; // uint64
    private static final int LOCATION_ADDRESS = 3; // uint64
    private static final int LOCATION_LINE = 4; // repeated Line
    private static final int LINE_FUNCTION_ID = 1; // uint64
    private static final int LINE_LINE = 2; // int64
    private static final int FUNCTION_ID = 1; // uint64
    private static final int FUNCTION_NAME = 2; // int64
    private static final int FUNCTION_SYSTEM_NAME = 3; // int64
    private static final int FUNCTION_FILENAME = 4; // int64
    private static final int FUNCTION_START_LINE = 5; // int64

    // Every string, function, and PC location is emitted once
    // and thereafter referred to by its identifier, a Long.
    private final Map<String, Long> stringIDs = new HashMap<>();
    private final Map<Long, Long> functionIDs = new HashMap<>(); // key is "address" of function
    private final Map<Long, Long> locationIDs = new HashMap<>(); // key is "address" of PC location

    // Returns the ID of the specified string,
    // emitting a pprof string record the first time it is encountered.
    private long getStringID(String s) throws IOException {
      Long i = stringIDs.putIfAbsent(s, Long.valueOf(stringIDs.size()));
      if (i == null) {
        enc.writeString(PROFILE_STRING_TABLE, s);
        return stringIDs.size() - 1L;
      }
      return i;
    }

    // Returns the ID of a StarlarkCallable for use in Line.FunctionId,
    // emitting a pprof Function record the first time fn is encountered.
    // The ID is the same as the function's logical address,
    // which is supplied by the caller to avoid the need to recompute it.
    private long getFunctionID(StarlarkCallable fn, long addr) throws IOException {
      Long id = functionIDs.get(addr);
      if (id == null) {
        id = addr;

        Location loc = fn.getLocation();
        String filename = loc.file(); // TODO(adonovan): make WORKSPACE-relative
        String name = fn.getName();
        if (name.equals("<toplevel>")) {
          name = filename;
        }

        long nameID = getStringID(name);

        ByteArrayOutputStream fun = new ByteArrayOutputStream();
        CodedOutputStream funEnc = CodedOutputStream.newInstance(fun);
        funEnc.writeUInt64(FUNCTION_ID, id);
        funEnc.writeInt64(FUNCTION_NAME, nameID);
        funEnc.writeInt64(FUNCTION_SYSTEM_NAME, nameID);
        funEnc.writeInt64(FUNCTION_FILENAME, getStringID(filename));
        funEnc.writeInt64(FUNCTION_START_LINE, (long) loc.line());
        funEnc.flush();
        enc.writeByteArray(PROFILE_FUNCTION, fun.toByteArray());

        functionIDs.put(addr, id);
      }
      return id;
    }

    // Returns the ID of the location denoted by fr,
    // emitting a pprof Location record the first time it is encountered.
    // For Starlark frames, this is the Frame pc.
    private long getLocationID(Debug.Frame fr) throws IOException {
      StarlarkCallable fn = fr.getFunction();
      // fnAddr identifies a function as a whole.
      int fnAddr = System.identityHashCode(fn); // very imperfect

      // pcAddr identifies the current program point.
      //
      // For now, this is the same as fnAddr, because
      // we don't track the syntax node currently being
      // evaluated. Statement-level profile information
      // in the leaf function (displayed by 'pprof list <fn>')
      // is thus unreliable for now.
      long pcAddr = fnAddr;
      if (fn instanceof StarlarkFunction) {
        // TODO(adonovan): when we use a byte code representation
        // of function bodies, mix the program counter fr.pc into fnAddr.
        // TODO(adonovan): even cleaner: treat each function's byte
        // code segment as its own Profile.Mapping, indexed by pc.
        //
        // pcAddr = (pcAddr << 16) ^ fr.pc;
      }

      Long id = locationIDs.get(pcAddr);
      if (id == null) {
        id = pcAddr;

        ByteArrayOutputStream line = new ByteArrayOutputStream();
        CodedOutputStream lineenc = CodedOutputStream.newInstance(line);
        lineenc.writeUInt64(LINE_FUNCTION_ID, getFunctionID(fn, fnAddr));
        lineenc.writeInt64(LINE_LINE, (long) fr.getLocation().line());
        lineenc.flush();

        ByteArrayOutputStream loc = new ByteArrayOutputStream();
        CodedOutputStream locenc = CodedOutputStream.newInstance(loc);
        locenc.writeUInt64(LOCATION_ID, id);
        locenc.writeUInt64(LOCATION_ADDRESS, pcAddr);
        locenc.writeByteArray(LOCATION_LINE, line.toByteArray());
        locenc.flush();
        enc.writeByteArray(PROFILE_LOCATION, loc.toByteArray());

        locationIDs.put(pcAddr, id);
      }
      return id;
    }
  }
}
