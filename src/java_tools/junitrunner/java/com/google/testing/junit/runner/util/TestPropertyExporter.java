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

import static com.google.testing.junit.runner.util.TestPropertyRunnerIntegration.getCallbackForThread;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;

import java.util.Map;

/**
 * Exports test properties to the test XML.
 */
public class TestPropertyExporter {
  /*
   * The global {@code TestPropertyExporter}, which writes the properties into
   * the test XML if the test is running from the command line.<p>
   *
   * If you have test infrastructure that needs to export properties, consider
   * injecting an instance of {@code TestPropertyExporter}. Your tests can
   * use one of the static methods in this class to create a fake instance.
   */
  public static final TestPropertyExporter INSTANCE = new TestPropertyExporter(
      new DefaultCallback());

  // Set to 1000 so that it will play nice with code that doesn't use exportRepeatedProperty
  // yet.
  public static final int INITIAL_INDEX_FOR_REPEATED_PROPERTY = 1000;

  private final Callback callback;

  /**
   * Creates a fake {@code TestPropertyExporter} instance, storing values
   * in the passed-in map.
   *
   * @param backingMap Map to use to store values
   * @return exporter instance
   */
  public static TestPropertyExporter createFake(final Map<String, String> backingMap) {
    return createFake(new Callback() {

      private final Multiset<String> repeatedPropertyNames = HashMultiset.create();

      @Override public void exportProperty(String name, String value) {
        backingMap.put(name, value);
      }

      @Override
      public String exportRepeatedProperty(String name, String value) {
        String propertyName = getRepeatedPropertyName(name);
        backingMap.put(propertyName, value);
        return propertyName;
      }

      private String getRepeatedPropertyName(String name) {
        int index = repeatedPropertyNames.add(name, 1) + INITIAL_INDEX_FOR_REPEATED_PROPERTY;
        return name + index;
      }
    });
  }

  /**
   * Creates a fake {@code TestPropertyExporter} instance, passing values
   * to the passed-in callback.
   *
   * @param callback Callback to use when values are exported
   * @return exporter instance
   */
  public static TestPropertyExporter createFake(final Callback callback) {
    return new TestPropertyExporter(callback);
  }

  protected TestPropertyExporter(Callback callback) {
    this.callback = callback;
  }

  /**
   * Exports a property to the test runner. This method is a no-op unless called
   * by the thread running the current test.
   *
   * @param name The property name.
   * @param value The property value.
   * @throws IllegalArgumentException if the name is not a valid name
   */
  public void exportProperty(String name, String value) {
    callback.exportProperty(name, value);
  }

  /**
   * Exports a property to the test runner by adding the value to the list of values for the
   * given property name.
   * When the properties get written to the XML, each name will have a numeric value appended to it
   * that is guaranteed to be unique for the given test case.
   * This method is a no-op unless called by the thread running the current test.
   *
   * @param name The property name.
   * @param value The property value.
   * @return the name of the property that was exported
   * @throws IllegalArgumentException if the name is not a valid name
   */
  public String exportRepeatedProperty(String name, String value) {
    return callback.exportRepeatedProperty(name, value);
  }

  /**
   * Callback that is used to store test properties.
   */
  public interface Callback {

    /**
     * Export the property.
     *
     * @param name The property name.
     * @param value The property value.
     */
    void exportProperty(String name, String value);

    /**
     * Export the property with an incrementing numeric suffix.
     *
     * @param name The property name.
     * @param value The property value.
     * @return the name of the property that was exported
     */
    String exportRepeatedProperty(String name, String value);
  }


  /**
   * Default callback implementation.
   * Calls the test runner to write the property to the XML.
   */
  private static class DefaultCallback implements Callback {

    @Override
    public void exportProperty(String name, String value) {
      getCallbackForThread().exportProperty(name, value);
    }

    @Override
    public String exportRepeatedProperty(String name, String value) {
      return getCallbackForThread().exportRepeatedProperty(name, value);
    }
  }
}
