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

package com.google.testing.junit.runner.junit4;

import com.google.testing.junit.runner.internal.junit4.JUnit4TestXmlListener;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;
import org.junit.runner.notification.RunListener;

/**
 * A factory that supplies a {@link RunListener} from a {@link JUnit4TestXmlListener}.
 */
public final class XmlListenerFactory implements Factory<RunListener> {
  private final Supplier<JUnit4TestXmlListener> implSupplier;

  public XmlListenerFactory(Supplier<JUnit4TestXmlListener> implSupplier) {
    assert implSupplier != null;
    this.implSupplier = implSupplier;
  }

  @Override
  public RunListener get() {
    RunListener runListener = JUnit4RunnerModule.xmlListener(implSupplier.get());
    assert runListener != null;
    return runListener;
  }

  public static Factory<RunListener> create(Supplier<JUnit4TestXmlListener> implSupplier) {
    return new XmlListenerFactory(implSupplier);
  }
}
