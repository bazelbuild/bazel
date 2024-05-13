// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.disk;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.HashSet;
import javax.annotation.Nullable;

/**
 * A wrapper around the SQLite C API.
 *
 * <p>Exposes only the subset of the SQLite C API required by Bazel.
 */
public final class Sqlite {

  static {
    JniLoader.loadJni();
  }

  private Sqlite() {}

  /**
   * Opens a connection to a database, creating it in an empty state if it doesn't yet exist.
   *
   * @param path the path to the database
   * @throws IOException if an error occurred while opening the connection
   */
  public static Connection newConnection(Path path) throws IOException {
    checkNotNull(path);
    return new Connection(path);
  }

  /**
   * A connection to a database.
   *
   * <p>This class is *not* thread-safe. A single connection should be used by a single thread. Use
   * separate connections to the same database for multithreaded access.
   */
  public static final class Connection implements AutoCloseable {

    // C pointer to the `sqlite3` handle. Zero means the connection has been closed.
    private long connPtr;

    // Tracks open statements so they can also be closed when the connection is closed.
    private final HashSet<Statement> openStatements = new HashSet<>();

    private Connection(Path path) throws IOException {
      this.connPtr = openConn(path.getPathString());
    }

    /**
     * Closes the connection, rendering it unusable.
     *
     * <p>As a convenience, every {@link Statement} associated with the connection is also closed,
     * with any errors silently ignored.
     *
     * <p>Multiple calls have no further effect.
     *
     * @throws IOException if an error occurred while closing the connection
     */
    @Override
    public void close() throws IOException {
      if (connPtr != 0) {
        // SQLite won't let us close the connection before first closing associated statements.
        for (Statement stmt : ImmutableList.copyOf(openStatements)) {
          stmt.close();
        }
        closeConn(connPtr);
        connPtr = 0;
      }
    }

    /**
     * Creates a new {@link Statement}.
     *
     * @param sql a string containing a single SQL statement
     * @throws IOException if the string contains multiple SQL statements, or the single SQL
     *     statement could not be parsed and validated
     */
    public Statement newStatement(String sql) throws IOException {
      checkState(connPtr != 0, "newStatement() called in invalid state");
      Statement stmt = new Statement(this, sql);
      openStatements.add(stmt);
      return stmt;
    }

    /**
     * Executes a statement not expected to return a result.
     *
     * <p>For statements expected to return a result, or statements that will be executed multiple
     * times, use {@link #newStatement}.
     *
     * @throws IOException if the string contains multiple SQL statements; or the single SQL
     *     statement could not be parsed and validated; or the statement returned a non-empty
     *     result; or an execution error occurred
     */
    public void executeUpdate(String sql) throws IOException {
      try (Statement stmt = new Statement(this, sql)) {
        stmt.executeUpdate();
      }
    }
  }

  /**
   * A prepared statement.
   *
   * <p>Provides a facility to bind values to parameters and execute the statement by calling one of
   * {@link #executeQuery} or {@link #executeUpdate}. The same statement may be executed multiple
   * times, with its parameters possibly bound to different values, but there can be at most one
   * ongoing execution at a time.
   *
   * <p>Parameters that haven't been bound or whose binding has been cleared behave as null.
   */
  public static final class Statement implements AutoCloseable {

    // The connection owning this statement.
    private Connection conn;

    // C pointer to te `sqlite3_stmt` handle. Zero means the statement has been closed.
    private long stmtPtr;

    // The result of current execution, or null if no execution is ongoing.
    @Nullable private Result currentResult;

    private Statement(Connection conn, String sql) throws IOException {
      this.conn = conn;
      stmtPtr = prepareStmt(conn.connPtr, sql);
      if (stmtPtr == -1) {
        // Special value returned when a multi-statement string is detected.
        throw new IOException("unsupported multi-statement string");
      }
    }

    /**
     * Closes the statement, rendering it unusable.
     *
     * <p>A {@link Result} previously returned by {@link #executeQuery} is also closed, with any
     * error silently ignored.
     *
     * <p>Multiple calls have no additional effect.
     */
    @Override
    public void close() {
      if (stmtPtr != 0) {
        if (currentResult != null) {
          try {
            currentResult.close();
          } catch (IOException e) {
            // Intentionally ignored: an error always pertains to a particular execution, and should
            // only be reported when Result#close is called directly.
          }
        }
        try {
          finalizeStmt(stmtPtr);
        } catch (IOException e) {
          // Cannot occur since the statement has been reset by Result#close.
          throw new IllegalStateException("unexpected exception thrown by finalize", e);
        }
        conn.openStatements.remove(this);
        conn = null;
        stmtPtr = 0;
      }
    }

    /**
     * Binds a long value to the statement's i-th parameter, counting from 1.
     *
     * <p>The binding remains in effect until it is cleared or another value is bound to the same
     * parameter.
     *
     * <p>Must not be called after {@link #executeQuery} returns a {@link Result} and before the
     * {@link Result} is closed.
     */
    public void bindLong(int i, long val) throws IOException {
      checkState(stmtPtr != 0 && currentResult == null, "bindLong() called in invalid state");
      bindStmtLong(stmtPtr, i, val);
    }

    /**
     * Binds a double value to the statement's i-th parameter, counting from 1.
     *
     * <p>The binding remains in effect until it is cleared or another value is bound to the same
     * parameter.
     *
     * <p>Must not be called after {@link #executeQuery} returns a {@link Result} and before the
     * {@link Result} is closed.
     */
    public void bindDouble(int i, double val) throws IOException {
      checkState(stmtPtr != 0 && currentResult == null, "bindDouble() called in invalid state");
      bindStmtDouble(stmtPtr, i, val);
    }

    /**
     * Binds a non-null string value to the statement's i-th parameter, counting from 1.
     *
     * <p>The binding remains in effect until it is cleared or another value is bound to the same
     * parameter.
     *
     * <p>Must not be called after {@link #executeQuery} returns a {@link Result} and before the
     * {@link Result} is closed.
     */
    public void bindString(int i, String val) throws IOException {
      checkState(stmtPtr != 0 && currentResult == null, "bindString() called in invalid state");
      checkNotNull(val);
      bindStmtString(stmtPtr, i, val);
    }

    /**
     * Clears the i-th binding.
     *
     * <p>Must not be called after {@link #executeQuery} returns a {@link Result} and before the
     * {@link Result} is closed.
     */
    public void clearBinding(int i) throws IOException {
      checkState(stmtPtr != 0 && currentResult == null, "clearBinding() called in invalid state");
      clearStmtBinding(stmtPtr, i);
    }

    /**
     * Clears all bindings.
     *
     * <p>Must not be called after {@link #executeQuery} returns a {@link Result} and before the
     * {@link Result} is closed.
     */
    public void clearBindings() throws IOException {
      checkState(stmtPtr != 0 && currentResult == null, "clearBindings() called in invalid state");
      clearStmtBindings(stmtPtr);
    }

    /**
     * Executes a statement expected to return a result.
     *
     * <p>Execution doesn't actually start until the first call to {@link Result#next}.
     *
     * <p>Must not be called again until the returned {@link Result} has been closed.
     */
    public Result executeQuery() {
      checkState(stmtPtr != 0 && currentResult == null, "executeQuery() called in invalid state");
      currentResult = new Result(this);
      return currentResult;
    }

    /**
     * Executes a statement not expected to return a result.
     *
     * <p>Must not be called after {@link #executeQuery} until the returned {@link Result} has been
     * closed.
     *
     * @throws IOException if the statement returned a non-empty result or an execution error
     *     occurred
     */
    public void executeUpdate() throws IOException {
      checkState(stmtPtr != 0 && currentResult == null, "executeUpdate() called in invalid state");
      currentResult = new Result(this);
      try {
        if (currentResult.next()) {
          throw new IOException("unexpected non-empty result");
        }
      } finally {
        currentResult.close();
      }
    }
  }

  /**
   * The result of executing a statement.
   *
   * <p>Acts as a cursor to iterate over result rows and obtain the corresponding column values. The
   * cursor is initially positioned before the first row. A call to {@link #next} moves the cursor
   * to the next row, returning {@code false} once no more results are available. If a call to
   * {@link #next} returns {@code true}, the getter methods may be called to retrieve the column
   * values for the current row.
   */
  public static final class Result implements AutoCloseable {

    // The statement owning this result.
    private Statement stmt;

    enum State {
      START, // next() not yet called
      CONTINUE, // last call to next() returned true
      DONE, // last call to next() returned false, but close() not yet called
      ERROR, // last call to next() threw, but close() not yet called
      CLOSED // close() was called
    }

    private State state = State.START;

    private Result(Statement stmt) {
      this.stmt = stmt;
    }

    /**
     * Closes the result, rendering it unusable.
     *
     * <p>Multiple calls have no additional effect.
     *
     * @throws IOException if an error occurred while finishing execution
     */
    @Override
    public void close() throws IOException {
      if (state != State.CLOSED) {
        try {
          resetStmt(stmt.stmtPtr);
        } catch (IOException e) {
          // Some statements may throw an error only after the result has been fully consumed.
          // However, if the error has already been thrown by next(), don't throw it again.
          if (state != State.ERROR) {
            throw e;
          }
        } finally {
          stmt.currentResult = null;
          stmt = null;
          state = State.CLOSED;
        }
      }
    }

    /**
     * Advances the cursor the next row.
     *
     * <p>Must not be called further after {@code false} is returned or an exception is thrown.
     *
     * @return whether another row was available
     * @throws IOException if an error occurred while executing the statement
     */
    public boolean next() throws IOException {
      checkState(state == State.START || state == State.CONTINUE, "next() called in invalid state");

      try {
        boolean available = stepStmt(stmt.stmtPtr);
        state = available ? State.CONTINUE : State.DONE;
        return available;
      } catch (IOException e) {
        state = State.ERROR;
        throw e;
      }
    }

    /**
     * Returns whether the i-th column for the current result row is null.
     *
     * <p>WARNING: this must be called before any of the getter methods. Calling a getter method may
     * cause a conversion to occur, after which the return value of this method is unspecified.
     */
    public boolean isNull(int i) throws IOException {
      checkState(state == State.CONTINUE, "isNull() called in invalid state");
      return columnIsNull(stmt.stmtPtr, i);
    }

    /**
     * Reads the i-th column for the current result row, starting from 0, as a long.
     *
     * <p>If the column is not of long type, a conversion occurs. In particular, null is converted
     * to 0.
     *
     * <p>Must not be called unless the last call to {@link #next} returned {@code true}.
     */
    public long getLong(int i) throws IOException {
      checkState(state == State.CONTINUE, "getLong() called in invalid state");
      return columnLong(stmt.stmtPtr, i);
    }

    /**
     * Reads the i-th column for the current result row, starting from 0, as a double.
     *
     * <p>If the column is not of double type, a conversion occurs. In particular, null is converted
     * to 0.0.
     *
     * <p>Must not be called unless the last call to {@link #next} returned {@code true}.
     */
    public double getDouble(int i) throws IOException {
      checkState(state == State.CONTINUE, "getDouble() called in invalid state");
      return columnDouble(stmt.stmtPtr, i);
    }

    /**
     * Reads the i-th column for the current result row, starting from 0, as a string.
     *
     * <p>If the column is not of string type, a conversion occurs. In particular, null is converted
     * to the empty string.
     *
     * <p>Must not be called unless the last call to {@link #next} returned {@code true}.
     */
    public String getString(int i) throws IOException {
      checkState(state == State.CONTINUE, "getString() called in invalid state");
      return columnString(stmt.stmtPtr, i);
    }
  }

  private static native long openConn(String path) throws IOException;

  private static native void closeConn(long conn) throws IOException;

  private static native long prepareStmt(long conn, String sql) throws IOException;

  private static native void bindStmtLong(long stmt, int i, long value) throws IOException;

  private static native void bindStmtDouble(long stmt, int i, double value) throws IOException;

  private static native void bindStmtString(long stmt, int i, String value) throws IOException;

  private static native void clearStmtBinding(long stmt, int i) throws IOException;

  private static native void clearStmtBindings(long stmt) throws IOException;

  private static native boolean stepStmt(long stmt) throws IOException;

  private static native boolean columnIsNull(long stmt, int i) throws IOException;

  private static native long columnLong(long stmt, int i) throws IOException;

  private static native double columnDouble(long stmt, int i) throws IOException;

  private static native String columnString(long stmt, int i) throws IOException;

  private static native void resetStmt(long stmt) throws IOException;

  private static native void finalizeStmt(long stmt) throws IOException;
}
