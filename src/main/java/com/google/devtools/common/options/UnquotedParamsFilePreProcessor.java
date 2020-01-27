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
package com.google.devtools.common.options;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * A {@link ParamsFilePreProcessor} that processes a parameter file using the {@code
 * com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType.UNQUOTED} format. This
 * format assumes each parameter is on a separate line and does not perform any special handling on
 * non-newline whitespace or special characters.
 */
public class UnquotedParamsFilePreProcessor extends ParamsFilePreProcessor {

  public UnquotedParamsFilePreProcessor(FileSystem fs) {
    super(fs);
  }

  @Override
  protected List<String> parse(Path paramsFile) throws IOException {
    return Files.readAllLines(paramsFile, UTF_8);
  }
}
