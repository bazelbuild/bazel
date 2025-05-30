// Copyright 2022 The Bazel Authors. All rights reserved.
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
// limitations under the License.package com.google.devtools.build.lib.runtime.commands;
package com.google.devtools.build.lib.runtime.commands;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.events.InputFileEvent;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;

/**
 * Reads the query for query, cquery and aquery using the --query_file option or from the residue of
 * the command line.
 */
public final class QueryOptionHelper {

  public static String readQuery(
      CommonQueryOptions queryOptions,
      OptionsParsingResult options,
      CommandEnvironment env,
      boolean allowEmptyQuery)
      throws QueryException {
    String query = "";
    if (!options.getResidue().isEmpty()) {
      if (!queryOptions.queryFile.isEmpty()) {
        throw new QueryException(
            "Command-line query and --query_file cannot both be specified",
            Query.Code.QUERY_FILE_WITH_COMMAND_LINE_EXPRESSION);
      }
      query = Joiner.on(' ').join(options.getResidue());
    } else if (!queryOptions.queryFile.isEmpty()) {
      // Works for absolute or relative query file.
      Path residuePath = env.getWorkingDirectory().getRelative(queryOptions.queryFile);
      try {
        env.getEventBus()
            .post(InputFileEvent.create(/* type= */ "query_file", residuePath.getFileSize()));
        query = new String(FileSystemUtils.readContent(residuePath), ISO_8859_1);
      } catch (IOException unused) {
        throw new QueryException(
            "I/O error reading from " + residuePath.getPathString(),
            Query.Code.QUERY_FILE_READ_FAILURE);
      }
    } else {
      // When querying for the state of Skyframe, it's possible to omit the query expression.
      if (!allowEmptyQuery) {
        throw new QueryException(
            String.format(
                "missing query expression. Type '%s help query' for syntax and help",
                env.getRuntime().getProductName()),
            Query.Code.COMMAND_LINE_EXPRESSION_MISSING);
      }
    }
    return query;
  }

  private QueryOptionHelper() {}
}
