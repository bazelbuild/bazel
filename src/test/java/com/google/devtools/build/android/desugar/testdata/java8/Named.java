// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.testdata.java8;

/** Desugar test interface to test precedence of inherited methods over default methods. */
public interface Named {
  default String name() {
    return getClass().getSimpleName();
  }

  /** Base class defining {@link #name} without implementing {@link Named}. */
  static class ExplicitNameBase {
    private final String name;

    public ExplicitNameBase(String name) {
      this.name = name;
    }

    public String name() {
      return name;
    }
  }

  /** Class whose base class implements {@link #name}. */
  public static class ExplicitName extends ExplicitNameBase implements Named {
    public ExplicitName(String name) {
      super(name);
    }
  }

  /** Class that explicitly defers to the default method in {@link Named}. */
  public static class DefaultName extends ExplicitNameBase implements Named {
    public DefaultName() {
      super(null);
    }

    @Override
    public String name() {
      return Named.super.name() + "-once";
    }
  }

  /** Subclass of {@link DefaultName} that uses {@code super} as well. */
  public static class DefaultNameSubclass extends DefaultName {
    @Override
    public String name() {
      return super.name() + "-twice";
    }
  }

  /** Base class that declares {@link #name} abstract. */
  abstract static class AbstractNameBase {
    public abstract String name();
  }

  /**
   * Class that inherits {@link #name} abstract so subclasses must implement it despite default
   * method in implemented interface.
   */
  public abstract static class AbstractName extends AbstractNameBase implements Named {}
}
