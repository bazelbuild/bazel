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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.ShutdownMethod;
import com.google.devtools.build.lib.server.RPCService.UnknownCommandException;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.RecordingOutErr;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Just makes sure the RPC service understands commands.
 */
@RunWith(JUnit4.class)
public class RPCServiceTest {

  private ServerCommand helloWorldCommand = new ServerCommand() {
    @Override
    public int exec(List<String> args, OutErr outErr, LockingMode lockingMode,
        String clientDescription, long firstContactTime) {
      outErr.printOut("Heelllloo....");
      outErr.printErr("...world!");
      return 42;
    }
    @Override
    public ShutdownMethod shutdown() {
      return ShutdownMethod.NONE;
    }
  };

  private RPCService service =
      new RPCService(helloWorldCommand);

  @Test
  public void testUnknownCommandException() {
    try {
      service.executeRequest(Arrays.asList("unknown"), new RecordingOutErr(), 0);
      fail();
    } catch (UnknownCommandException e) {
      // success
    } catch (Exception e){
      fail();
    }
  }

  @Test
  public void testCommandGetsExecuted() throws Exception {
    RecordingOutErr outErr = new RecordingOutErr();
    int exitStatus = service.executeRequest(Arrays.asList("blaze"), outErr, 0);

    assertEquals(42, exitStatus);
    assertEquals("Heelllloo....", outErr.outAsLatin1());
    assertEquals("...world!", outErr.errAsLatin1());
  }

  @Test
  public void testDelimitation() throws Exception {
    final List<String> savedArgs = new ArrayList<>();

    RPCService service =
        new RPCService(new ServerCommand() {
            @Override
            public int exec(List<String> args, OutErr outErr, LockingMode lockingMode,
                String clientDescription, long firstContactTime) {
              savedArgs.addAll(args);
              return 0;
            }
            @Override
            public ShutdownMethod shutdown() {
              return ShutdownMethod.NONE;
            }
          });

    List<String> args = Arrays.asList("blaze", "", " \n", "", "", "", "foo");
    service.executeRequest(args, new RecordingOutErr(), 0);
    assertEquals(args.subList(1, args.size()),
                 savedArgs);
  }

  @Test
  public void testShutdownState() throws Exception {
    assertEquals(ShutdownMethod.NONE, service.getShutdown());
    service.shutdown(ShutdownMethod.CLEAN);
    assertEquals(ShutdownMethod.CLEAN, service.getShutdown());
    service.shutdown(ShutdownMethod.EXPUNGE);
    assertEquals(ShutdownMethod.CLEAN, service.getShutdown());
  }

  @Test
  public void testExpungeShutdown() throws Exception {
    assertEquals(ShutdownMethod.NONE, service.getShutdown());
    service.shutdown(ShutdownMethod.EXPUNGE);
    assertEquals(ShutdownMethod.EXPUNGE, service.getShutdown());
  }

  @Test
  public void testCommandFailsAfterShutdown() throws Exception {
    RecordingOutErr outErr = new RecordingOutErr();
    service.shutdown(ShutdownMethod.CLEAN);
    try {
      service.executeRequest(Arrays.asList("blaze"), outErr, 0);
      fail();
    } catch (IllegalStateException e) {
      assertThat(e).hasMessage("Received request after shutdown.");
      /* Make sure it does not execute the command! */
      assertThat(outErr.outAsLatin1()).isEmpty();
      assertThat(outErr.errAsLatin1()).isEmpty();
    }
  }

}
