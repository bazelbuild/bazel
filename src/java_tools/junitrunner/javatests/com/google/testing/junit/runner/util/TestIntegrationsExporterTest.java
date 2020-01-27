// Copyright 2011 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

import static org.mockito.Mockito.verify;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;

/** Tests for {@link TestIntegrationsExporter}. */
@RunWith(MockitoJUnitRunner.class)
public class TestIntegrationsExporterTest {
  @Mock private TestIntegrationsExporter.Callback mockCallback;
  private TestIntegrationsExporter.Callback previousCallback;

  @Before
  public void setThreadCallback() throws Exception {
    previousCallback = TestIntegrationsRunnerIntegration.setTestCaseForThread(mockCallback);
  }

  @After
  public void restorePreviousThreadCallback() {
    TestIntegrationsRunnerIntegration.setTestCaseForThread(previousCallback);
  }

  @Test
  public void testExportTestIntegration() {
    final TestIntegration testIntegration =
        TestIntegration.builder()
            .setContactEmail("test@testmail.com")
            .setComponentId("1234")
            .setName("Test")
            .setUrl("testurl")
            .setDescription("Test description.")
            .setForegroundColor("white")
            .setBackgroundColor("rgb(47, 122, 243)")
            .build();

    TestIntegrationsExporter.INSTANCE.newTestIntegration(testIntegration);
    verify(mockCallback).exportTestIntegration(testIntegration);
  }
}
