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
package com.google.devtools.build.android;

import com.google.devtools.build.android.ParsedAndroidData.KeyValueConsumer;

/**
 * A group of {@link KeyValueConsumer}s for each DataValue type.
 *
 * This class acts as a parameter object for organizing the common grouping of consumer instances.
 */
class KeyValueConsumers {
  static KeyValueConsumers of(
      KeyValueConsumer<DataKey, DataResource> overwritingConsumer,
      KeyValueConsumer<DataKey, DataResource> combiningConsumer,
      KeyValueConsumer<DataKey, DataAsset> assetConsumer) {
    return new KeyValueConsumers(overwritingConsumer, combiningConsumer, assetConsumer);
  }

  final KeyValueConsumer<DataKey, DataResource> overwritingConsumer;
  final KeyValueConsumer<DataKey, DataResource> combiningConsumer;
  final KeyValueConsumer<DataKey, DataAsset> assetConsumer;

  private KeyValueConsumers(
      KeyValueConsumer<DataKey, DataResource> overwritingConsumer,
      KeyValueConsumer<DataKey, DataResource> combiningConsumer,
      KeyValueConsumer<DataKey, DataAsset> assetConsumer) {
    this.overwritingConsumer = overwritingConsumer;
    this.combiningConsumer = combiningConsumer;
    this.assetConsumer = assetConsumer;
  }
}
