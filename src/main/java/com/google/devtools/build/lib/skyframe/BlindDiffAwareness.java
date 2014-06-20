// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.vfs.ModifiedFileSet;

/**
 * Dummy class that implements blind diff awareness and always says everything is modified.
 */
public class BlindDiffAwareness implements DiffAwareness {

  @Override
  public ModifiedFileSet getDiff() {
    return ModifiedFileSet.EVERYTHING_MODIFIED;
  }

  @Override
  public boolean canStillBeUsed() {
    // We return false here so that a more sophisticated diff awareness strategy has a chance to be
    // chosen, if it becomes applicable. If there are problems with a more sophisticated strategy,
    // we'll use BlindDiffAwareness but we don't want to be stuck with it forever.
    return false;
  }
}
