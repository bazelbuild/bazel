package com.google.devtools.build.lib.remote.common;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

import static com.google.common.truth.Truth.assertThat;

@RunWith(JUnit4.class)
public class BulkTransferExceptionTest {

  @Test
  public void shouldProvideGenericMessageIfNoAddedException() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    assertThat(bulkTransferException.getMessage()).isEqualTo("Unknown error during bulk transfer");
  }

  @Test
  public void shouldPreserveMessageAsIsFromSingleException() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException("Failure Type A"));
    assertThat(bulkTransferException.getMessage()).isEqualTo("Failure Type A");
  }

  @Test
  public void shouldSortAndRemoveDuplicatesWhenAggregatingMessages() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException("Failure Type B"));
    bulkTransferException.add(new IOException("Failure Type A"));
    bulkTransferException.add(new IOException("Failure Type B"));
    assertThat(bulkTransferException.getMessage()).isEqualTo("Multiple errors during bulk transfer:\n" +
        "Failure Type A\n" +
        "Failure Type B");
  }

  @Test
  public void shouldProvideGenericMessageIfOnlyNullMessages() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException());
    assertThat(bulkTransferException.getMessage()).isEqualTo("Unknown error during bulk transfer");
  }

  @Test
  public void shouldIgnoreNullMessagesWhenGettingMessage() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException("Failure Type A"));
    bulkTransferException.add(new IOException());
    assertThat(bulkTransferException.getMessage()).isEqualTo("Failure Type A");
  }
}


