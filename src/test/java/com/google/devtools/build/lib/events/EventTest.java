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
package com.google.devtools.build.lib.events;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.events.Event.ProcessOutput;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkThread.PrintHandler;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.SyntaxError;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.InOrder;

/** Tests for {@link Event}. */
@RunWith(JUnit4.class)
public class EventTest {

  @Test
  public void eventKindMessage() {
    Event event = Event.of(EventKind.WARNING, "myMessage");

    assertThat(event.getMessage()).isEqualTo("myMessage");
    assertThat(event.getKind()).isEqualTo(EventKind.WARNING);
  }

  @Test
  public void eventMessageEncoding() {
    String message = "Bazel \u1f33f";

    Event stringEvent = Event.of(EventKind.WARNING, message);
    Event stringEvent2 = Event.of(EventKind.WARNING, "Bazel \u1f33f");
    assertThat(stringEvent.getMessage()).isEqualTo(message);
    assertThat(stringEvent.getMessageBytes()).isEqualTo(message.getBytes(UTF_8));

    Event byteArrayEvent = Event.of(EventKind.WARNING, message.getBytes(UTF_8));
    Event byteArrayEvent2 = Event.of(EventKind.WARNING, "Bazel \u1f33f".getBytes(UTF_8));
    assertThat(byteArrayEvent.getMessage()).isEqualTo(message);
    assertThat(byteArrayEvent.getMessageBytes()).isEqualTo(message.getBytes(UTF_8));

    new EqualsTester()
        .addEqualityGroup(stringEvent, stringEvent2)
        .addEqualityGroup(byteArrayEvent, byteArrayEvent2)
        .testEquals();
  }

  @Test
  public void eventLocationSensitiveToString() {
    String file = "/path/to/workspace/my/sample/path.txt";
    Location location = Location.fromFileLineColumn(file, 3, 4);
    Event event = Event.of(EventKind.WARNING, "myMessage", Location.class, location);

    assertThat(event.getLocation()).isEqualTo(location);
    assertThat(event.toString()).isEqualTo("WARNING " + file + ":3:4: myMessage");
  }

  @Test
  public void messageReference() throws Exception {
    byte[] messageBytes = "message".getBytes(UTF_8);
    Event event = Event.of(EventKind.WARNING, messageBytes);
    assertThat(event.getMessageBytes()).isEqualTo(messageBytes);
  }

  @Test
  public void noProperties() {
    Event event = Event.of(EventKind.WARNING, "myMessage");
    assertThat(event.getProperty(Object.class)).isNull();
    assertThat(event).isEqualTo(Event.of(EventKind.WARNING, "myMessage"));
  }

  @Test
  public void oneProperty() {
    Event event = Event.of(EventKind.WARNING, "myMessage", String.class, "myProperty");
    assertThat(event.getProperty(Object.class)).isNull();
    assertThat(event.getProperty(String.class)).isEqualTo("myProperty");
    assertThat(event)
        .isEqualTo(Event.of(EventKind.WARNING, "myMessage", String.class, "myProperty"));
  }

  @Test
  public void withAddedProperty() {
    Event event = Event.of(EventKind.WARNING, "myMessage", String.class, "myProperty");
    Location location = Location.fromFileLineColumn("file", 1, 2);
    Event twoPropertyEvent = event.withProperty(Location.class, location);

    assertThat(event).isNotSameInstanceAs(twoPropertyEvent);
    assertThat(event).isNotEqualTo(twoPropertyEvent);
    assertThat(event.getProperty(String.class)).isEqualTo("myProperty");
    assertThat(event.getProperty(Location.class)).isNull();
    assertThat(twoPropertyEvent.getProperty(String.class)).isEqualTo("myProperty");
    assertThat(twoPropertyEvent.getProperty(Location.class)).isSameInstanceAs(location);
  }

  @Test
  public void withReplacedProperty() {
    Location location = Location.fromFileLineColumn("file", 1, 2);
    Event event =
        Event.of(EventKind.WARNING, "myMessage")
            .withProperty(String.class, "myProperty")
            .withProperty(Location.class, location);
    Event replacedPropertyEvent = event.withProperty(String.class, "yourProperty");

    assertThat(event).isNotSameInstanceAs(replacedPropertyEvent);
    assertThat(event).isNotEqualTo(replacedPropertyEvent);
    assertThat(event.getProperty(String.class)).isEqualTo("myProperty");
    assertThat(event.getProperty(Location.class)).isSameInstanceAs(location);
    assertThat(replacedPropertyEvent.getProperty(String.class)).isEqualTo("yourProperty");
    assertThat(replacedPropertyEvent.getProperty(Location.class)).isSameInstanceAs(location);
  }

  @Test
  public void withRemovedProperty() {
    Location location = Location.fromFileLineColumn("file", 1, 2);
    Event event =
        Event.of(EventKind.WARNING, "myMessage")
            .withProperty(String.class, "myProperty")
            .withProperty(Location.class, location);
    Event removedPropertyEvent = event.withProperty(Location.class, null);

    assertThat(event).isNotSameInstanceAs(removedPropertyEvent);
    assertThat(event).isNotEqualTo(removedPropertyEvent);
    assertThat(event.getProperty(String.class)).isEqualTo("myProperty");
    assertThat(event.getProperty(Location.class)).isSameInstanceAs(location);
    assertThat(removedPropertyEvent.getProperty(String.class)).isEqualTo("myProperty");
    assertThat(removedPropertyEvent.getProperty(Location.class)).isNull();
    assertThat(removedPropertyEvent.withProperty(Location.class, null))
        .isSameInstanceAs(removedPropertyEvent);
  }

  @Test
  public void propertyOrderAgnostic() {
    Location location = Location.fromFileLineColumn("file", 1, 2);
    Event stringFirstEvent =
        Event.of(EventKind.WARNING, "myMessage")
            .withProperty(String.class, "myProperty")
            .withProperty(Location.class, location);
    Event locationFirstEvent =
        Event.of(EventKind.WARNING, "myMessage")
            .withProperty(Location.class, location)
            .withProperty(String.class, "myProperty");
    new EqualsTester().addEqualityGroup(stringFirstEvent, locationFirstEvent).testEquals();
  }

  @Test
  public void withTag() {
    Event event = Event.of(EventKind.WARNING, "myMessage").withTag("myTag");
    assertThat(event.getTag()).isEqualTo("myTag");
    assertThat(event.withTag("myTag")).isSameInstanceAs(event);

    Event withoutTag = event.withTag(null);
    assertThat(withoutTag.getTag()).isNull();
    assertThat(withoutTag.withTag(null)).isSameInstanceAs(withoutTag);
  }

  @Test
  public void tagIsSameAsStringProperty() {
    assertThat(Event.of(EventKind.WARNING, "myMessage", String.class, "myProperty").getTag())
        .isEqualTo("myProperty");
  }

  @Test
  public void testWithProcessOutput() throws Exception {
    String stdoutPath = "/stdout";
    byte[] stdout = "some stdout output".getBytes(UTF_8);
    String stderrPath = "/stderr";
    byte[] stderr = "some stderr error".getBytes(UTF_8);

    ProcessOutput testProcessOutput =
        new ProcessOutput() {
          @Override
          public String getStdOutPath() {
            return stdoutPath;
          }

          @Override
          public long getStdOutSize() {
            return stdout.length;
          }

          @Override
          public byte[] getStdOut() {
            return stdout;
          }

          @Override
          public String getStdErrPath() {
            return stderrPath;
          }

          @Override
          public long getStdErrSize() {
            return stderr.length;
          }

          @Override
          public byte[] getStdErr() {
            return stderr;
          }
        };

    Event event = Event.of(EventKind.WARNING, "myMessage");
    Event eventWithProcessOutput = event.withProcessOutput(testProcessOutput);

    assertThat(eventWithProcessOutput).isNotEqualTo(event);
    assertThat(event.getProcessOutput()).isNull();
    assertThat(eventWithProcessOutput.getProcessOutput()).isNotNull();

    assertThat(eventWithProcessOutput.getStdOut()).isEqualTo(stdout);
    assertThat(eventWithProcessOutput.getProcessOutput().getStdOut()).isEqualTo(stdout);
    assertThat(eventWithProcessOutput.getProcessOutput().getStdOutSize()).isEqualTo(stdout.length);

    assertThat(eventWithProcessOutput.getStdErr()).isEqualTo(stderr);
    assertThat(eventWithProcessOutput.getProcessOutput().getStdErr()).isEqualTo(stderr);
    assertThat(eventWithProcessOutput.getProcessOutput().getStdErrSize()).isEqualTo(stderr.length);
  }

  @Test
  public void replayEventsOn() {
    ImmutableList<Event> events =
        ImmutableList.of(
            Event.of(EventKind.INFO, "someInfo"), Event.of(EventKind.WARNING, "someWarning"));

    EventHandler mock = mock(EventHandler.class);

    Event.replayEventsOn(mock, events);

    InOrder inOrder = inOrder(mock);
    inOrder.verify(mock).handle(events.get(0));
    inOrder.verify(mock).handle(events.get(1));
  }

  @Test
  public void replaySyntaxErrorsOn() {
    Location location1 = Location.fromFileLineColumn("someFile", 3, 4);
    Location location2 = Location.fromFileLineColumn("someOtherFile", 5, 6);
    ImmutableList<SyntaxError> syntaxErrors =
        ImmutableList.of(
            new SyntaxError(location1, "message1"), new SyntaxError(location2, "message2"));

    EventHandler mock = mock(EventHandler.class);
    Event.replayEventsOn(mock, syntaxErrors);

    InOrder inOrder = inOrder(mock);
    inOrder
        .verify(mock)
        .handle(Event.error(syntaxErrors.get(0).location(), syntaxErrors.get(0).message()));
    inOrder
        .verify(mock)
        .handle(Event.error(syntaxErrors.get(1).location(), syntaxErrors.get(1).message()));
  }

  @Test
  public void debugPrintHandler() {
    EventHandler mockHandler = mock(EventHandler.class);
    PrintHandler printHandler = Event.makeDebugPrintHandler(mockHandler);
    StarlarkThread starlarkThread =
        StarlarkThread.createTransient(Mutability.create(), StarlarkSemantics.DEFAULT);

    printHandler.print(starlarkThread, "someMessage");

    verify(mockHandler).handle(Event.debug(Location.BUILTIN, "someMessage"));
  }
}
