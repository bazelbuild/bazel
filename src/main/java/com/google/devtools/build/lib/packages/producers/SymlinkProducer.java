// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.producers;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.function.Consumer;

/**
 * Looks up {@link FileValue} for a {@link FileKey} which is guaranteed to be a symlink.
 *
 * <p>Used when {@link PatternWithWildcardProducer} handles {@link
 * com.google.devtools.build.lib.skyframe.DirectoryListingValue}. For any {@link
 * com.google.devtools.build.lib.vfs.Dirent} which is a {@code SYMLINK}, computes its {@link
 * FileValue} to know the target path.
 *
 * <p>When handling each {@code DirectoryListingValue}, multiple {@link SymlinkProducer}s can be
 * created so that {@link com.google.devtools.build.skyframe.state.Driver} is able to query the
 * symlink dirents in a batch. All symlink dirents {@link FileValue} will be collected in an array
 * list and the runAfter method should be executed only once. So {@link SymlinkProducer} does not
 * expect any runAfter {@link StateMachine} to be passed in.
 *
 * <p>If the {@link FileValue} from skyframe shows that this is not a symlink, accepts an {@link
 * InconsistentFilesystemException} which will be bubbled up.
 */
final class SymlinkProducer implements StateMachine, Consumer<SkyValue> {

  interface ResultSink {
    void acceptSymlinkFileValue(FileValue symlinkValue, FileKey symlinkKey);

    void acceptInconsistentFilesystemException(InconsistentFilesystemException exception);
  }

  // -------------------- Input --------------------
  private final FileKey symlinkKey;

  // -------------------- Output --------------------
  private final ResultSink resultSink;

  SymlinkProducer(FileKey symlinkKey, ResultSink resultSink) {
    this.symlinkKey = symlinkKey;
    this.resultSink = resultSink;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.lookUp(symlinkKey, (Consumer<SkyValue>) this);
    return DONE;
  }

  @Override
  public void accept(SkyValue skyValue) {
    Preconditions.checkState(skyValue instanceof FileValue);
    FileValue symlinkValue = (FileValue) skyValue;

    if (!symlinkValue.isSymlink()) {
      resultSink.acceptInconsistentFilesystemException(
          new InconsistentFilesystemException(
              "readdir and stat disagree about whether "
                  + symlinkKey.argument().asPath()
                  + " is a symlink."));
      return;
    }

    resultSink.acceptSymlinkFileValue(symlinkValue, symlinkKey);
  }
}
