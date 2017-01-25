// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import static com.google.common.base.Preconditions.checkState;

import com.android.dx.cf.direct.DirectClassFile;
import com.android.dx.dex.file.DexFile;
import java.io.IOException;

class MergingDexer {

  // NB: The following two constants are copied from com.android.dx.command.dexer.Main

  /**
   * Maximum number of methods added during dexing:
   * <ul>
   * <li>Array.newInstance may be added by RopperMachine,
   * <li>ArrayIndexOutOfBoundsException.<init> may be added by EscapeAnalysis
   * </ul>
   */
  private static final int MAX_METHOD_ADDED_DURING_DEX_CREATION = 2;

  /** Maximum number of fields added during dexing: &lt;primitive types box class&gt;.TYPE. */
  private static final int MAX_FIELD_ADDED_DURING_DEX_CREATION = 9;

  private final int maxNumberOfIdxPerDex;
  private final Dexing dexing;
  private final DexFileAggregator dest;
  private final boolean multidex;
  private DexFile currentShard;

  public MergingDexer(
      Dexing dexing,
      DexFileAggregator dest,
      boolean multidex,
      int maxNumberOfIdxPerDex) {
    this.dexing = dexing;
    this.dest = dest;
    this.multidex = multidex;
    this.maxNumberOfIdxPerDex = maxNumberOfIdxPerDex;
    currentShard = dexing.newDexFile();
  }

  public MergingDexer add(DirectClassFile classFile) throws IOException {
    if (multidex && !currentShard.isEmpty()) {
      // NB: This code is copied from com.android.dx.command.dexer.Main

      // Calculate max number of indices this class will add to the
      // dex file.
      // The possibility of overloading means that we can't easily
      // know how many constant are needed for declared methods and
      // fields. We therefore make the simplifying assumption that
      // all constants are external method or field references.

      int constantPoolSize = classFile.getConstantPool().size();
      int maxMethodIdsInClass = constantPoolSize + classFile.getMethods().size()
              + MAX_METHOD_ADDED_DURING_DEX_CREATION;
      int maxFieldIdsInClass = constantPoolSize + classFile.getFields().size()
              + MAX_FIELD_ADDED_DURING_DEX_CREATION;

      if (maxNumberOfIdxPerDex < getCurrentShardFieldCount() + maxFieldIdsInClass
          || maxNumberOfIdxPerDex < getCurrentShardMethodCount() + maxMethodIdsInClass) {
        // For simplicity just start a new shard to fit the given file
        // Don't bother with waiting for a later file that might fit the old shard as in the extreme
        // we'd have to wait until the end to write all shards.
        rotateDexFile();
      }
    }

    dexing.addToDexFile(currentShard, classFile);
    return this;
  }

  public void finish() throws IOException {
    if (currentShard != null && !currentShard.isEmpty()) {
      dest.add(currentShard);
    }
    currentShard = null;
  }

  public void flush() throws IOException {
    checkState(multidex);
    if (!currentShard.isEmpty()) {
      dest.add(currentShard);
    }
    currentShard = dexing.newDexFile();
  }

  private void rotateDexFile() throws IOException {
    checkState(multidex);
    checkState(!currentShard.isEmpty());
    // We could take advantage here of knowing that currentShard is "full" and force dest to just
    // write it out instead of it trying to merge currentShard with other .dex files.
    // We're not doing that for the moment because that can cause problems when processing a
    // main_dex_list: if dest's currentShard still contains classes from main_dex_list then writing
    // our currentShard (with classes not from main dex list) separately would cause dest to write
    // classes.dex with our currentShard and classes from main_dex_list to end up in classes2.dex or
    // later, unless we prevent that case somehow (e.g., by knowing that order matters when writing
    // the first shard).
    dest.add(currentShard);
    currentShard = dexing.newDexFile();
  }

  private int getCurrentShardMethodCount() {
    return currentShard.getMethodIds().items().size();
  }

  private int getCurrentShardFieldCount() {
    return currentShard.getFieldIds().items().size();
  }
}
