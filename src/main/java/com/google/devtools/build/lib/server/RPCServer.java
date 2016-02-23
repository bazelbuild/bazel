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

package com.google.devtools.build.lib.server;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.server.RPCService.UnknownCommandException;
import com.google.devtools.build.lib.server.signal.InterruptSignalHandler;
import com.google.devtools.build.lib.unix.LocalClientSocket;
import com.google.devtools.build.lib.unix.LocalServerSocket;
import com.google.devtools.build.lib.unix.LocalSocketAddress;
import com.google.devtools.build.lib.unix.NativePosixFiles;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.ThreadUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.StreamMultiplexer;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * An RPCServer server is a Java object that sits and waits for RPC requests
 * (the sit-and-wait is implemented in {@link #serve()}).  These requests
 * arrive via UNIX file sockets. The RPCServer then calls the application
 * (which implements ServerCommand) to handle the request. (Since the Blaze
 * server may need to stat hundreds of directories during initialization, this
 * is a significant speedup.)  The server thread will terminate after idling
 * for a user-specified time.
 *
 * Note: If you are contemplating to call into the RPCServer from
 * within Java, consider using the {@link RPCService} class instead.
 */
// TODO(bazel-team): Signal handling.
// TODO(bazel-team): Gives clients status information when the server is busy. One
// way to do this is to put the server status in a file (pid, the current
// target, etc) in the server directory. Alternatively, we can have a separate
// thread taking care of the server socket and put the information into socket
// handshakes.
// TODO(bazel-team): Use Reporter for server-side messages.
public final class RPCServer {

  private final Clock clock;
  private final RPCService rpcService;
  private final LocalServerSocket serverSocket;
  private final long maxIdleMillis;
  private final long statusCheckMillis;
  private final Path serverDirectory;
  private final Path workspaceDir;
  private static final Logger LOG = Logger.getLogger(RPCServer.class.getName());
  private volatile boolean lameDuck;

  private static final long STATUS_CHECK_PERIOD_MILLIS = 1000 * 60; // 1 minute.
  private static final Splitter NULLTERMINATOR_SPLITTER = Splitter.on('\0');

  /**
   * Create a new server instance. After creating the server, you can start it
   * by calling the {@link #serve()} method.
   *
   * @param clock The clock to take time measurements
   * @param rpcService The underlying service object, which takes
   *                           care of dispatching to the {@link ServerCommand}
   *                           instances, as requests arrive.
   * @param maxIdleMillis      The maximum time the server will wait idly.
   * @param statusCheckPeriodMillis How long to wait between system status checks.
   * @param serverDirectory    Directory to put file socket and pid files, etc.
   * @param workspaceDir The workspace. Used solely to ensure it persists.
   * @throws IOException
   */
  public RPCServer(Clock clock, RPCService rpcService,
                   long maxIdleMillis, long statusCheckPeriodMillis,
                   Path serverDirectory, Path workspaceDir)
      throws IOException {
    this.clock = clock;
    this.rpcService = rpcService;
    this.maxIdleMillis = maxIdleMillis;
    this.statusCheckMillis = statusCheckPeriodMillis;
    this.serverDirectory = serverDirectory;
    this.workspaceDir = workspaceDir;

    this.serverSocket = openServerSocket();
    serverSocket.setSoTimeout(Math.min(maxIdleMillis, statusCheckMillis));
    lameDuck = false;
  }

  /**
   * Create a new server instance. After creating the server, you can start it
   * by calling the {@link #serve()} method.
   *
   * @param clock The clock to take time measurements
   * @param rpcService The underlying service object, which takes
   *                           care of dispatching to the {@link ServerCommand}
   *                           instances, as requests arrive.
   * @param maxIdleMillis      The maximum time the server will wait idly.
   * @param serverDirectory    Directory to put file socket and pid files, etc.
   * @param workspaceDir       The workspace. Used solely to ensure it persists.
   * @throws IOException
   */
  public RPCServer(Clock clock, RPCService rpcService,
      long maxIdleMillis, Path serverDirectory, Path workspaceDir)
      throws IOException {
    this(clock, rpcService, maxIdleMillis, STATUS_CHECK_PERIOD_MILLIS,
        serverDirectory, workspaceDir);
  }

  private static void printStack(IOException e) {
    /*
     * Hopefully this never happens. It's not very nice to just write this
     * to the user's console, but I'm not sure what better choice we have.
     */
    StringWriter err = new StringWriter();
    PrintWriter printErr = new PrintWriter(err);
    printErr.println("=======[BLAZE SERVER: ENCOUNTERED IO EXCEPTION]=======");
    e.printStackTrace(printErr);
    printErr.println("=====================================================");
    LOG.severe(err.toString());
  }

  /**
   * Wait on a socket for business (answer requests). Note that this
   * method won't return until the server shuts down.
   */
  public void serve() {
    // Register the signal handler.
    final AtomicBoolean inAction = new AtomicBoolean(false);
    final AtomicBoolean allowingInterrupt = new AtomicBoolean(true);
    final AtomicLong cmdNum = new AtomicLong();
    final Thread mainThread = Thread.currentThread();
    final Object interruptLock = new Object();

    InterruptSignalHandler sigintHandler = new InterruptSignalHandler() {
        @Override
        public void run() {
          LOG.severe("User interrupt");

          // Only interrupt during actions - otherwise we may end up setting the interrupt bit
          // at the end of a build and responding to it at the beginning of the subsequent build.
          synchronized (interruptLock) {
            if (allowingInterrupt.get()) {
              mainThread.interrupt();
            }
          }

          Runnable interruptWatcher = new Runnable() {
            @Override
            public void run() {
              try {
                long originalCmd = cmdNum.get();
                Thread.sleep(10 * 1000);
                if (inAction.get() && cmdNum.get() == originalCmd) {
                  // We're still operating on the same command.
                  // Interrupt took too long.
                  ThreadUtils.warnAboutSlowInterrupt();
                }
              } catch (InterruptedException e) {
                // Ignore.
              }
            }
          };

          if (inAction.get()) {
            Thread interruptWatcherThread =
                new Thread(interruptWatcher, "interrupt-watcher-" + cmdNum);
            interruptWatcherThread.setDaemon(true);
            interruptWatcherThread.start();
          }
        }
      };

    try {
      while (!lameDuck) {
        try {
          IdleServerTasks idleChecker = new IdleServerTasks(workspaceDir);
          idleChecker.idle();
          RequestIo requestIo;

          long startTime = clock.currentTimeMillis();
          while (true) {
            try {
              allowingInterrupt.set(true);
              Socket socket = serverSocket.accept();
              long firstContactTime = clock.currentTimeMillis();
              requestIo = new RequestIo(socket, firstContactTime);
              break;
            } catch (SocketTimeoutException e) {
              long idleTime = clock.currentTimeMillis() - startTime;
              if (lameDuck) {
                closeServerSocket();
                return;
              } else if (idleTime > maxIdleMillis ||
                  (idleTime > statusCheckMillis && !idleChecker.continueProcessing(idleTime))) {
                enterLameDuck();
              }
            }
          }
          idleChecker.busy();


          List<String> request = null;
          try {
            request = extractRequest(requestIo);
            cmdNum.incrementAndGet();
            inAction.set(true);
            if (request != null) {
              executeRequest(request, requestIo);
            }
          } finally {
            inAction.set(false);
            // Don't reset interruption unless we executed a request. Otherwise this is just a
            // ping from the client verifying our existence, in which case we should retain the
            // interrupt status for the subsequent request.
            if (request != null) {
              synchronized (interruptLock) {
                allowingInterrupt.set(false);
                Thread.interrupted(); // clears thread interrupted status
              }
            }
            requestIo.shutdown();
            if (rpcService.isShutdown()) {
              return;
            }
          }
        } catch (IOException e) {
          if (e.getMessage().equals("Broken pipe")) {
            LOG.info("Connection to the client lost: "
                           + e.getMessage());
          } else {
            // Other cases: print the stack for debugging.
            printStack(e);
          }
        }
      }
    } finally {
      rpcService.shutdown();
      LOG.info("Logging finished");
      sigintHandler.uninstall();
    }
  }

  private void closeServerSocket() {
    LOG.info("Closing serverSocket.");
    try {
      serverSocket.close();
    } catch (IOException e) {
      printStack(e);
    }

    if (!lameDuck) {
      try {
        getSocketPath().delete();
      } catch (IOException e) {
        printStack(e);
      }
    }
  }

  /**
   * Allow one last request to be serviced.
   */
  private void enterLameDuck() {
    lameDuck = true;
    try {
      getSocketPath().delete();
    } catch (IOException e) {
      e.printStackTrace();
    }
    serverSocket.setSoTimeout(1);
  }

  /**
   * Returns the path of the socket file to be used.
   */
  public Path getSocketPath() {
    return serverDirectory.getRelative("server.socket");
  }

  /**
   * Ensures no other server is running for the current socket file.  This
   * guarantees that no two servers are running against the same output
   * directory.
   *
   * @throws IOException if another server holds the lock for the socket file.
   */
  public static void ensureExclusiveAccess(Path socketFile) throws IOException {
    LocalSocketAddress address =
        new LocalSocketAddress(socketFile.getPathFile());
    if (socketFile.exists()) {
      try {
        new LocalClientSocket(address).close();
      } catch (IOException e) {
        // The previous server process is dead--unlink the file:
        socketFile.delete();
        return;
      }
      // TODO(bazel-team): (2009) Read the previous server's pid from the "hello" message
      // and add it to the message.
      throw new IOException("Socket file " + socketFile.getPathString()
                            + " is locked by another server");
    }
  }

  /**
   * Schedule the specified file for (attempted) deletion at JVM exit.
   */
  private static void deleteAtExit(final Path socketFile, final boolean deleteParent) {
    Runtime.getRuntime().addShutdownHook(new Thread() {
        @Override
        public void run() {
          try {
            socketFile.delete();
            if (deleteParent) {
              socketFile.getParentDirectory().delete();
            }
          } catch (IOException e) {
            printStack(e);
          }
        }
      });
  }

  /**
   * Opens a UNIX local server socket.
   * @throws IOException if the socket file is used by another server or can
   * not be made exclusive.
   */
  private LocalServerSocket openServerSocket() throws IOException {
    // This is the "well known" socket path via which the server is found...
    Path socketFile = getSocketPath();

    // ...but it may have a name that's too long for AF_UNIX, in which case we
    // make it a symlink to /tmp/something.  This typically only happens in
    // tests where the --output_base is beneath a very deep temp dir.
    // (All this extra complexity is just used in tests... *sigh*).
    if (socketFile.toString().length() >= 108) { // = UNIX_PATH_MAX
      Path socketLink = socketFile;
      String tmpDir = System.getProperty("blaze.rpcserver.tmpdir", "/tmp");
      socketFile = createTempSocketDirectory(socketFile.getRelative(tmpDir)).
          getRelative("server.socket");
      LOG.info("Using symlinked socket at " + socketFile);

      socketLink.delete(); // Remove stale symlink, if any.
      socketLink.createSymbolicLink(socketFile);

      deleteAtExit(socketLink, /*deleteParent=*/false);
      deleteAtExit(socketFile, /*deleteParent=*/true);
    } else {
      deleteAtExit(socketFile, /*deleteParent=*/false);
    }

    ensureExclusiveAccess(socketFile);

    // We create the server.pid file strictly before binding the socket.
    // The client only accesses the pid file after connecting to the socket
    // which ensures that it gets the correct pid value.
    Path pidFile = serverDirectory.getRelative("server.pid");
    deleteAtExit(pidFile, /*deleteParent=*/ false);
    try {
      pidFile.delete();
    } catch (IOException e) {
      // Ignore.
    }
    pidFile.createSymbolicLink(new PathFragment(String.valueOf(OsUtils.getpid())));

    LocalServerSocket serverSocket = new LocalServerSocket();
    serverSocket.bind(new LocalSocketAddress(socketFile.getPathFile()));
    NativePosixFiles.chmod(socketFile.getPathFile(), 0600);  // Lock it down.
    serverSocket.listen(/*backlog=*/50);
    return serverSocket;
  }

  // Atomically create a new directory in the (assumed sticky) /tmp directory for use with a
  // Unix domain socket. The directory will be mode 0700. Retries indefinitely until it
  // succeeds.
  private static Path createTempSocketDirectory(Path tempDir) {
    Random random = new Random();
    while (true) {
      Path socketDir = tempDir.getRelative(String.format("blaze-%d", random.nextInt()));
      try {
        if (socketDir.createDirectory()) {
          // Make sure it's private; unfortunately, createDirectory() doesn't take a mode
          // argument.
          socketDir.chmod(0700);
          return socketDir; // Created.
        }
        // Already existed; try again.
      } catch (IOException e) {
        // Failed; try again.
      }
    }
  }

  /**
   * Read a string in platform default encoding and split it into a list of
   * NUL-separated words.
   *
   * <p>Blaze consistently uses the platform default encoding (defined in
   * blaze.cc) to interface with Unix APIs.
   */
  private static List<String> readRequest(InputStream input) throws IOException {
    byte[] inputBytes = ByteStreams.toByteArray(input);
    if (inputBytes.length == 0) {
      return null;
    }
    String s = new String(inputBytes, Charset.defaultCharset());
    return ImmutableList.copyOf(NULLTERMINATOR_SPLITTER.split(s));
  }

  private static List<String> extractRequest(RequestIo requestIo) throws IOException {
    List<String> request = readRequest(requestIo.in);
    if (request == null) {
      LOG.info("Short-circuiting empty request");
      return null;
    }
    return request;
  }

  private void executeRequest(List<String> request, RequestIo requestIo) {
    Preconditions.checkNotNull(request);
    int exitStatus = 2;
    try {
      exitStatus = rpcService.executeRequest(request, requestIo.requestOutErr,
              requestIo.firstContactTime);
      LOG.info("Finished executing request");
    } catch (UnknownCommandException e) {
      requestIo.requestOutErr.printErrLn("SERVER ERROR: " + e.getMessage());
      LOG.severe("SERVER ERROR: " + e.getMessage());
    } catch (Exception e) {
      // Stacktrace for unknown exception.
      StringWriter trace = new StringWriter();
      e.printStackTrace(new PrintWriter(trace, true));
      requestIo.requestOutErr.printErr("SERVER ERROR: " + trace);
      LOG.severe("SERVER ERROR: " + trace);
    }

    if (rpcService.isShutdown()) {
      // In case of shutdown, disable the listening socket *before* we write
      // the last part of the response.  Otherwise, a sufficiently fast client
      // could read the response and exit, and a new client could make a
      // connection to this server, which is still in the listening state, even
      // though it is about to shut down imminently.
      closeServerSocket();
    }

    requestIo.writeExitStatus(exitStatus);
  }

  /**
   * Because it's a little complicated, this class factors out all the IO Hook
   * up we need per request, that is, in
   * {@link RPCServer#executeRequest(List, RequestIo)}.
   * It's unfortunately complicated, so it's explained here.
   */
  private static class RequestIo {

    // Used by the client code
    private final InputStream in;
    private final OutErr requestOutErr;
    private final OutputStream controlChannel;

    // just used by this class to keep the state around
    private final Socket requestSocket;
    private final OutputStream requestOut;
    private final long firstContactTime;

    RequestIo(Socket requestSocket, long firstContactTime) throws IOException {
      this.requestSocket = requestSocket;
      this.firstContactTime = firstContactTime;
      this.in = requestSocket.getInputStream();
      this.requestOut = requestSocket.getOutputStream();

      // We encode the response sent to the client with a multiplexer so
      // we can send three streams (out / err / control) over one wire stream
      // (requestOut).
      StreamMultiplexer multiplexer = new StreamMultiplexer(requestOut);

      // We'll be writing control messages (exit code + out of date message)
      // to this control channel.
      controlChannel = multiplexer.createControl();

      // This is the outErr part of the multiplexed output.
      requestOutErr = OutErr.create(multiplexer.createStdout(),
                                    multiplexer.createStderr());
      // We hook up System.out / System.err to our IO object. Stuff written to
      // System.out / System.err will show up on the user's screen, prefixed
      // with "System.out "/"System.err ".
      requestOutErr.addSystemOutErrAsSource();
    }

    public void writeExitStatus(int exitStatus) {
      // Make sure to flush the output / error streams prior to writing the exit status.
      // The client may stop reading that direction of the socket immediately upon reading the
      // exit code.
      flushOutErr();
      try {
        controlChannel.write(("" + exitStatus + "\n").getBytes(UTF_8));
        controlChannel.flush();
        LOG.info("" + exitStatus);
      } catch (IOException ignored) {
        // This exception is historically ignored.
      }
    }

    private void flushOutErr() {
      try {
        requestOutErr.getOutputStream().flush();
      } catch (IOException e) {
        printStack(e);
      }
      try {
        requestOutErr.getErrorStream().flush();
      } catch (IOException e) {
        printStack(e);
      }
    }

    public void shutdown() {
      try {
        requestOut.close();
      } catch (IOException e) {
        printStack(e);
      }
      try {
        in.close();
      } catch (IOException e) {
        printStack(e);
      }
      try {
        requestSocket.close();
      } catch (IOException e) {
        printStack(e);
      }
    }
  }

  /**
   * Creates and returns a new RPC server.
   * Use {@link RPCServer#serve()} to start the server.
   *
   * @param appCommand The application's ServerCommand implementation.
   * @param serverDirectory The directory for server-related files. The caller
   * must ensure the directory has been created.
   * @param workspaceDir The workspace, used solely to ensure it persists.
   * @param maxIdleSeconds The idle time in seconds after which the rpc
   * server will die unless it receives a request.
   */
  public static RPCServer newServerWith(Clock clock,
                                        ServerCommand appCommand,
                                        Path serverDirectory,
                                        Path workspaceDir,
                                        int maxIdleSeconds)
      throws IOException {
    if (!serverDirectory.exists()) {
      serverDirectory.createDirectory();
    }

    // Creates and starts the RPC server.
    RPCService service = new RPCService(appCommand);

    return new RPCServer(clock, service, maxIdleSeconds * 1000L,
                         serverDirectory, workspaceDir);
  }

}
