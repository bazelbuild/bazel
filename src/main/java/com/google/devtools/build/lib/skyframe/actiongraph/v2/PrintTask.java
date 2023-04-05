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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

import com.google.auto.value.AutoValue;
import com.google.protobuf.Message;
import javax.annotation.Nullable;

/**
 * Represent a task to be consumed by a {@link AqueryConsumingOutputHandler}.
 *
 * <p>We have separate Proto/TextProto subclasses to reduce some memory waste: we'll never need both
 * the fieldNumber and the messageLabel in a PrintTask.
 */
@SuppressWarnings("InterfaceWithOnlyStatics")
public interface PrintTask {

  /** A task for the proto format. */
  @AutoValue
  abstract class ProtoPrintTask implements PrintTask {
    @Nullable
    abstract Message message();

    abstract int fieldNumber();

    public static ProtoPrintTask create(Message message, int fieldNumber) {
      return new AutoValue_PrintTask_ProtoPrintTask(message, fieldNumber);
    }
  }

  /** A task for the textproto format. */
  @AutoValue
  abstract class TextProtoPrintTask implements PrintTask {
    @Nullable
    abstract Message message();

    abstract String messageLabel();

    public static TextProtoPrintTask create(Message message, String messageLabel) {
      return new AutoValue_PrintTask_TextProtoPrintTask(message, messageLabel);
    }
  }
}
