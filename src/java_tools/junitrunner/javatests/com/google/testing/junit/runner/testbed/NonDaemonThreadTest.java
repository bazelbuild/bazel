package com.google.testing.junit.runner.testbed;

import org.junit.Test;

public class NonDaemonThreadTest {

    @Test
    public void testNonDaemonThread() {
        Thread thread = new Thread(() -> {
            try {
                // Simulate some work with sleep
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        thread.setDaemon(false); // Set the thread as non-daemon
        thread.start();
    }

}
