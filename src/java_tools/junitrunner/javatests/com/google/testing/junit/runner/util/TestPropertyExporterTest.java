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

/**
 * Tests for {@link TestPropertyExporter}.
 */
@RunWith(MockitoJUnitRunner.class)
public class TestPropertyExporterTest {
  @Mock private TestPropertyExporter.Callback mockCallback;
  private TestPropertyExporter.Callback previousCallback;

  @Before
  public void setThreadCallback() throws Exception {
    previousCallback = TestPropertyRunnerIntegration.setTestCaseForThread(mockCallback);
  }

  @After
  public void restorePreviousThreadCallback() throws Exception {
    TestPropertyRunnerIntegration.setTestCaseForThread(previousCallback);
  }

  @Test
  public void testExportProperty() {
    TestPropertyExporter.INSTANCE.exportProperty("propertyName", "value");
    verify(mockCallback).exportProperty("propertyName", "value");
  }

  @Test
  public void testExportRepeatedProperty() {
    TestPropertyExporter.INSTANCE.exportRepeatedProperty("propertyName", "value");
    verify(mockCallback).exportRepeatedProperty("propertyName", "value");
  }

  @Test
  public void testExportProperty_emptyNameIsValid() {
    TestPropertyExporter.INSTANCE.exportProperty(" ", "value");
    verify(mockCallback).exportProperty(" ", "value");
  }

  @Test
  public void testExportRepeatedProperty_emptyNameIsValid() {
    TestPropertyExporter.INSTANCE.exportRepeatedProperty(" ", "value");
    verify(mockCallback).exportRepeatedProperty(" ", "value");
  }
}
