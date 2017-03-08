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

package com.google.testing.junit.runner;

import com.google.testing.junit.runner.model.AntXmlResultWriter;
import com.google.testing.junit.runner.model.XmlResultWriter;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;

/**
 * A factory that supplies {@link XmlResultWriter}.
 */
public final class ResultWriterFactory implements Factory<XmlResultWriter> {
  private final Supplier<AntXmlResultWriter> implSupplier;

  public ResultWriterFactory(Supplier<AntXmlResultWriter> implSupplier) {
    assert implSupplier != null;
    this.implSupplier = implSupplier;
  }

  @Override
  public XmlResultWriter get() {
    XmlResultWriter xmlResultWriter = BazelTestRunnerModule.resultWriter(implSupplier.get());
    assert xmlResultWriter != null;
    return xmlResultWriter;
  }

  public static Factory<XmlResultWriter> create(Supplier<AntXmlResultWriter> implSupplier) {
    return new ResultWriterFactory(implSupplier);
  }
}
