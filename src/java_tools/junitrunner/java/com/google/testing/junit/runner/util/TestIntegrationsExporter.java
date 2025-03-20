// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

import static com.google.testing.junit.runner.util.TestIntegrationsRunnerIntegration.getCallbackForThread;

/** Exports test TestIntegrations to the test XML. */
public class TestIntegrationsExporter {
  /*
   * The global {@code TestIntegrationsExporter}, which writes the properties into
   * the test XML if the test is running from the command line.
   *
   * <p>If you have test infrastructure that needs to export properties, consider
   * injecting an instance of {@code TestIntegrationsExporter}. Your tests can
   * use one of the static methods in this class to create a fake instance.
   */
  public static final TestIntegrationsExporter INSTANCE =
      new TestIntegrationsExporter(new DefaultCallback());

  private final Callback callback;

  protected TestIntegrationsExporter(Callback callback) {
    this.callback = callback;
  }

  public void newTestIntegration(TestIntegration testIntegration) {
    callback.exportTestIntegration(testIntegration);
  }

  /** Callback that is used to store TestIntegration in the model. */
  public interface Callback {
    /** Export the TestIntegration. */
    void exportTestIntegration(TestIntegration testIntegration);
  }

  /**
   * Default callback implementation. Calls the test runner model to write the external integrations
   * to the XML.
   */
  private static class DefaultCallback implements Callback {

    @Override
    public void exportTestIntegration(TestIntegration testIntegration) {
      getCallbackForThread().exportTestIntegration(testIntegration);
    }
  }
}
