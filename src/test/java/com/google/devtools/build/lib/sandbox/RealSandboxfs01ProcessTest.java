// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * {@inheritDoc}
 *
 * <p><b>Important:</b> If you modify any aspects of the golden data of these tests, you must run
 * the {@link RealSandboxfsProcessIntegrationTest} test against the version of sandboxfs tested here
 * to ensure that the golden data represents reality.
 */
@RunWith(JUnit4.class)
public class RealSandboxfs01ProcessTest extends BaseRealSandboxfsProcessTest {

  public RealSandboxfs01ProcessTest() {
    super("0.1.0", RealSandboxfsProcess.ALLOW_FLAG + "\n" + FAKE_MOUNT_POINT + "\n");
  }

  @Test
  public void testNoActivity() throws IOException {
    SandboxfsProcess process = createAndStartFakeSandboxfs(ImmutableList.of("Done"));
    String expectedRequests = "[]\n\n";
    verifyFakeSandboxfsExecution(process, expectedRequests);
  }

  @Test
  public void testCreateAndDestroySandboxes() throws IOException {
    SandboxfsProcess process =
        createAndStartFakeSandboxfs(ImmutableList.of("Done", "Done", "Done", "Done"));

    process.createSandbox("sandbox1", (mapper) -> {});
    process.createSandbox(
        "sandbox2",
        (mapper) -> mapper.map(PathFragment.create("/"), PathFragment.create("/some/path"), true));
    process.destroySandbox("sandbox1");
    String expectedRequests =
        "[]\n\n"
            + "[]\n\n"
            + "[{\"Map\":"
            + "{\"Mapping\":\"/sandbox2\",\"Target\":\"/some/path\",\"Writable\":true}}]\n\n"
            + "[{\"Unmap\":\"/sandbox1\"}]\n\n";

    verifyFakeSandboxfsExecution(process, expectedRequests);
  }
}
