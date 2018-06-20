package com.google.devtools.build.lib.integration.blackbox.framework.junit;

import java.util.concurrent.TimeoutException;
import org.junit.rules.TestWatcher;
import org.junit.rules.Timeout;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

public abstract class TimeoutTestWatcher extends TestWatcher {
  private String name;

  protected abstract long getTimeoutMillis();
  protected abstract boolean onTimeout();

  @Override
  protected void starting(Description description) {
    name = description.getMethodName();
  }

  @Override
  protected void finished(Description description) {
    name = null;
  }

  public String getName() {
    return name;
  }

  @Override
  public Statement apply(Statement base, Description description) {
    // we are using exception wrapping, because unfortunately JUnit's Timeout throws
    // java.util.Exception on timeout, which is hard to distinguish from other cases
    Statement wrapper = new Statement() {
      @Override
      public void evaluate() throws Throwable {
        try {
          base.evaluate();
        } catch (Throwable th) {
          throw new ExceptionWrapper(th);
        }
      }
    };

    return new Statement() {
      @Override
      public void evaluate() throws Throwable {
        try {
          new Timeout((int) getTimeoutMillis()).apply(wrapper, description).evaluate();
        } catch (ExceptionWrapper wrapper) {
          // original test exception
          throw wrapper.getCause();
        } catch (Exception e) {
          // timeout exception
          if (!onTimeout()) {
            throw new TimeoutException(e.getMessage());
          }
        }
      }
    };
  }

  private static class ExceptionWrapper extends Throwable {
    ExceptionWrapper(Throwable cause) {
      super(cause);
    }
  }
}
