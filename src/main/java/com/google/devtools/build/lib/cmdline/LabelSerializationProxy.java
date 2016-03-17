// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.cmdline;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectOutput;

/**
 * A serialization proxy for {@code Label}.
 */
public class LabelSerializationProxy implements Externalizable {

  private String labelString;

  public LabelSerializationProxy(String labelString) {
    this.labelString = labelString;
  }

  // For deserialization machinery.
  public LabelSerializationProxy() {
  }

  @Override
  public void writeExternal(ObjectOutput out) throws IOException {
    // Manual serialization gives us about a 30% reduction in size.
    out.writeUTF(labelString);
  }

  @Override
  public void readExternal(java.io.ObjectInput in) throws IOException {
    this.labelString = in.readUTF();
  }

  private Object readResolve() {
    return Label.parseAbsoluteUnchecked(labelString, false);
  }
}
