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

package com.google.devtools.build.lib.server;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.JavaClock;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FsApparatus;

import junit.framework.TestCase;

import java.io.File;
import java.util.List;

/**
 * Run a real RPC server on localhost, and talk to it using the testing
 * client.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
public class RPCServerTest extends TestCase {

  private static final long MAX_IDLE_MILLIS = 10000;
  private static final long HEALTH_CHECK_MILLIS = 1000 * 3;
  private static final String COMMAND_STDOUT = "Heelllloo....";
  private static final String COMMAND_STDERR = "...world!";

  private RPCServer server;
  private FsApparatus scratch = FsApparatus.newNative();
  private RecordingOutErr outErr = new RecordingOutErr();
  private Path serverDir;
  private Path workspaceDir;
  private RPCTestingClient client;
  private Thread serverThread = new Thread(){
    @Override
    public void run() {
      server.serve();
    }
  };

  private static final ServerCommand helloWorldCommand = new ServerCommand() {
    @Override
    public int exec(List<String> args, OutErr outErr, long firstContactTime) throws Exception {
      outErr.printOut(COMMAND_STDOUT);
      outErr.printErr(COMMAND_STDERR);
      return 42;
    }
    @Override
    public boolean shutdown() {
      return false;
    }
  };

  @Override
  protected void setUp() throws Exception {
    super.setUp();

    // Do not use `createUnixTempDir()` here since the file name that results is longer
    // than 108 characters, so cannot be used as local socket address.
    File file = File.createTempFile("scratch", ".tmp", new File("/tmp"));
    file.delete();
    file.mkdir();
    serverDir = this.scratch.dir(file.getAbsolutePath());

    workspaceDir = this.scratch.createUnixTempDir();
    workspaceDir.createDirectory();
    client = new RPCTestingClient(
        outErr, serverDir.getRelative("server.socket"));
    RPCService service = new RPCService(helloWorldCommand);
    server = new RPCServer(new JavaClock(), service, MAX_IDLE_MILLIS, HEALTH_CHECK_MILLIS,
        serverDir, workspaceDir);
    serverThread.start();
  }

  @Override
  protected void tearDown() throws Exception {
    serverThread.interrupt();
    serverThread.join();

    FileSystemUtils.deleteTree(serverDir);
    super.tearDown();
  }

  protected void testRequest(String request, int ret, String out, String err,
                             String control) throws Exception {
    assertEquals(new ServerResponse(control, ret), client.sendRequest(request));
    assertEquals(out, outErr.outAsLatin1());
    assertThat(outErr.errAsLatin1()).contains(err);
  }

  public void testUnknownCommand() throws Exception {
    testRequest("unknown", 2, "", "SERVER ERROR: Unknown command: unknown\n", "");
  }

  public void testEmptyBlazeCommand() throws Exception {
    testRequest("unknown", 2, "", "SERVER ERROR: Unknown command: unknown\n", "");
  }

  public void testWorkspaceDies() throws Exception {
    assertTrue(serverThread.isAlive());
    testRequest("blaze", 42, COMMAND_STDOUT, COMMAND_STDERR, "");
    Thread.sleep(HEALTH_CHECK_MILLIS * 2);
    assertTrue(serverThread.isAlive());

    assertTrue(workspaceDir.delete());
    Thread.sleep(HEALTH_CHECK_MILLIS * 2);
    assertFalse(serverThread.isAlive());
  }
}
