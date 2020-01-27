// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.testbed;

import com.google.common.base.Preconditions;

/**
 * A sample class that is under test by XmlOutputExercises.
 */
public class ExampleObject implements Comparable<ExampleObject> {
  private String data;

  public ExampleObject(String data) {
    this.data = Preconditions.checkNotNull(data);
  }

  public String getData() {
    return data;
  }

  public void setData(String data) {
    this.data = Preconditions.checkNotNull(data);
  }

  @Override
  public int compareTo(ExampleObject that) {
    return this.data.compareTo(that.data);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    ExampleObject that = (ExampleObject) o;
    return data.equals(that.data);
  }

  @Override
  public int hashCode() {
    return data.hashCode();
  }

  @Override
  public String toString() {
    return this.data;
  }
}
