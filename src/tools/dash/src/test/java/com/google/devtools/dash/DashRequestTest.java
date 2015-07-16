// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.dash;

import static com.google.common.truth.Truth.assertThat;
import static org.easymock.EasyMock.createMock;
import static org.easymock.EasyMock.expect;
import static org.easymock.EasyMock.replay;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.appengine.api.datastore.Blob;
import com.google.appengine.api.datastore.Entity;
import com.google.appengine.tools.development.testing.LocalDatastoreServiceTestConfig;
import com.google.appengine.tools.development.testing.LocalServiceTestHelper;
import com.google.devtools.build.lib.bazel.dash.DashProtos;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import javax.servlet.http.HttpServletRequest;

/**
 * Tests for {@link DashRequest}.
 */
@RunWith(JUnit4.class)
public class DashRequestTest {

  private final LocalServiceTestHelper helper =
      new LocalServiceTestHelper(new LocalDatastoreServiceTestConfig());
  private HttpServletRequest request;

  @Before
  public void setUp() {
    helper.setUp();
    request = createMock(HttpServletRequest.class);
  }

  @After
  public void tearDown() {
    helper.tearDown();
  }

  @Test
  public void testUriParsing() throws Exception {
    final String buildId = "3b9a81d9-0ed3-48d2-84c6-6296aecc21e6";
    final DashProtos.BuildData data = DashProtos.BuildData.newBuilder().setBuildId(buildId).build();
    expect(request.getRequestURI()).andReturn("/result/3b9a81d9-0ed3-48d2-84c6-6296aecc21e6");
    expect(request.getInputStream()).andReturn(new ProtoInputStream(data));
    replay(request);

    DashRequest dashRequest = new DashRequest(request);
    assertEquals(buildId, dashRequest.getBuildId());
    Entity entity = dashRequest.getEntity();
    assertEquals(buildId, entity.getProperty(DashRequest.BUILD_ID));
    assertEquals("result", entity.getProperty(DashRequest.PAGE_NAME));
    assertEquals(data, DashProtos.BuildData.parseFrom(
        ((Blob) entity.getProperty(DashRequest.BUILD_DATA)).getBytes()));
  }

  private void uriError(String uri) {
    expect(request.getRequestURI()).andReturn(uri).times(2);
    replay(request);

    try {
      new DashRequest(request);
      fail("Should have thrown");
    } catch (DashRequest.DashRequestException e) {
      assertThat(e.getMessage()).contains("Invalid URI pattern: " + uri);
    }
  }

  @Test
  public void testInvalidUuid() {
    uriError("/result/3b9a81d9-0ed3-48d2");
  }

  @Test
  public void testMissingPageName() {
    uriError("/3b9a81d9-0ed3-48d2-84c6-6296aecc21e6");
  }

}
