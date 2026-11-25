// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Target;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;

/** A query output format that can retain metadata for targets with unresolved hashes. */
public interface HashOutputFormatter extends StreamedFormatter {
  void writeHashedTargets(
      Iterable<Target> targets,
      LabelPrinter labelPrinter,
      Map<Label, ByteString> hashes,
      OutputStream out)
      throws IOException, InterruptedException;
}
