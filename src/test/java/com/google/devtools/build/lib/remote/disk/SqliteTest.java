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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.remote.disk.Sqlite.Connection;
import com.google.devtools.build.lib.remote.disk.Sqlite.Result;
import com.google.devtools.build.lib.remote.disk.Sqlite.SqliteException;
import com.google.devtools.build.lib.remote.disk.Sqlite.Statement;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SqliteTest {

  private Path dbPath;

  @Before
  public void setUp() throws Exception {
    dbPath = TestUtils.createUniqueTmpDir(null).getChild("tmp.db");
  }

  @After
  public void tearDown() throws Exception {
    try {
      dbPath.delete();
    } catch (IOException e) {
      // Intentionally ignored.
    }
  }

  @Test
  public void executeQuery_withEmptyResult_nextReturnsFalse() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT 1 LIMIT 0");
        Result result = stmt.executeQuery()) {
      assertThat(result.next()).isFalse();
    }
  }

  @Test
  public void executeQuery_withNonEmptyResult_nextReturnsTrue() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3");
        Result result = stmt.executeQuery()) {
      assertThat(result.next()).isTrue();
      assertThat(result.next()).isTrue();
      assertThat(result.next()).isTrue();
      assertThat(result.next()).isFalse();
    }
  }

  @Test
  public void executeQuery_getterCalledBeforeNextReturnsTrue_throws() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT 1");
        Result result = stmt.executeQuery()) {
      assertThrows(IllegalStateException.class, () -> result.isNull(0));
      assertThrows(IllegalStateException.class, () -> result.getLong(0));
      assertThrows(IllegalStateException.class, () -> result.getDouble(0));
      assertThrows(IllegalStateException.class, () -> result.getString(0));
    }
  }

  @Test
  public void executeQuery_getterCalledAfterNextReturnsFalse_throws() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT 1 LIMIT 0");
        Result result = stmt.executeQuery()) {
      assertThat(result.next()).isFalse();
      assertThrows(IllegalStateException.class, () -> result.isNull(0));
      assertThrows(IllegalStateException.class, () -> result.getLong(0));
      assertThrows(IllegalStateException.class, () -> result.getDouble(0));
      assertThrows(IllegalStateException.class, () -> result.getString(0));
    }
  }

  @Test
  public void executeQuery_prematureClose_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT 1");
        Result result = stmt.executeQuery()) {}
  }

  @Test
  public void executeQuery_gettersConvertNull() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT NULL, NULL, NULL");
        Result result = stmt.executeQuery()) {
      assertThat(result.next()).isTrue();
      assertThat(result.isNull(0)).isTrue();
      assertThat(result.getLong(0)).isEqualTo(0);
      assertThat(result.isNull(1)).isTrue();
      assertThat(result.getDouble(1)).isEqualTo(0.0);
      assertThat(result.isNull(2)).isTrue();
      assertThat(result.getString(2)).isEmpty();
    }
  }

  @Test
  public void executeQuery_onError_throws() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("CREATE TABLE tbl (id INTEGER)")) {
      try (Result result = stmt.executeQuery()) {
        assertThat(result.next()).isFalse();
      }
      try (Result result = stmt.executeQuery()) {
        SqliteException e = assertThrows(SqliteException.class, result::next);
        assertThat(e).hasMessageThat().contains("SQL logic error");
        assertThat(e.getCode()).isEqualTo(Sqlite.ERR_ERROR);
      }
    }
  }

  @Test
  public void executeUpdate_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath)) {
      try (Statement stmt = conn.newStatement("CREATE TABLE tbl (id INTEGER)")) {
        stmt.executeUpdate();
      }

      try (Statement stmt = conn.newStatement("SELECT COUNT(*) FROM tbl");
          Result result = stmt.executeQuery()) {
        assertThat(result.next()).isTrue();
      }
    }
  }

  @Test
  public void executeUpdate_oneShot_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath)) {
      conn.executeUpdate("CREATE TABLE tbl (id INTEGER)");

      try (Statement stmt = conn.newStatement("SELECT COUNT(*) FROM tbl");
          Result result = stmt.executeQuery()) {
        assertThat(result.next()).isTrue();
      }
    }
  }

  @Test
  public void executeUpdate_withParameters_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath)) {
      try (Statement stmt = conn.newStatement("CREATE TABLE tbl AS SELECT ?, ?, ?")) {
        stmt.bindLong(1, 42);
        stmt.bindDouble(2, 3.14);
        stmt.bindString(3, "hello");
        stmt.executeUpdate();
      }

      try (Statement stmt = conn.newStatement("SELECT * FROM tbl");
          Result result = stmt.executeQuery()) {
        assertThat(result.next()).isTrue();
        assertThat(result.isNull(0)).isFalse();
        assertThat(result.getLong(0)).isEqualTo(42);
        assertThat(result.isNull(1)).isFalse();
        assertThat(result.getDouble(1)).isEqualTo(3.14);
        assertThat(result.isNull(2)).isFalse();
        assertThat(result.getString(2)).isEqualTo("hello");
        assertThat(result.next()).isFalse();
      }
    }
  }

  @Test
  public void executeUpdate_withNonEmptyResult_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT NULL")) {}
  }

  @Test
  public void executeUpdate_onError_throws() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("CREATE TABLE tbl (id INTEGER)")) {
      stmt.executeUpdate();
      SqliteException e = assertThrows(SqliteException.class, stmt::executeUpdate);
      assertThat(e).hasMessageThat().contains("SQL logic error");
      assertThat(e.getCode()).isEqualTo(Sqlite.ERR_ERROR);
    }
  }

  @Test
  public void clearBinding_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT ?, ?")) {
      stmt.bindString(1, "abc");
      stmt.bindString(2, "def");

      stmt.clearBinding(1);
      try (Result result = stmt.executeQuery()) {
        assertThat(result.next()).isTrue();
        assertThat(result.isNull(0)).isTrue();
        assertThat(result.isNull(1)).isFalse();
        assertThat(result.next()).isFalse();
      }

      stmt.clearBinding(2);
      try (Result result = stmt.executeQuery()) {
        assertThat(result.next()).isTrue();
        assertThat(result.isNull(0)).isTrue();
        assertThat(result.isNull(1)).isTrue();
        assertThat(result.next()).isFalse();
      }
    }
  }

  @Test
  public void clearBindings_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT ?, ?")) {
      stmt.bindString(1, "abc");
      stmt.bindString(2, "def");
      stmt.clearBindings();

      try (Result result = stmt.executeQuery()) {
        assertThat(result.next()).isTrue();
        assertThat(result.isNull(0)).isTrue();
        assertThat(result.isNull(1)).isTrue();
        assertThat(result.next()).isFalse();
      }
    }
  }

  @Test
  public void newStatement_withInvalidStatement_throws() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath)) {
      SqliteException e =
          assertThrows(SqliteException.class, () -> conn.newStatement("i am not sql"));
      assertThat(e).hasMessageThat().contains("SQL logic error");
      assertThat(e.getCode()).isEqualTo(Sqlite.ERR_ERROR);
    }
  }

  @Test
  public void newStatement_withTrailingSemicolon_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT 1;")) {}
  }

  @Test
  public void newStatement_withMultipleStatements_throws() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath)) {
      IOException e =
          assertThrows(IOException.class, () -> conn.newStatement("SELECT 1; SELECT 2"));
      assertThat(e).hasMessageThat().contains("unsupported multi-statement string");
    }
  }

  @Test
  public void closeStatement_withOpenResult_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath);
        Statement stmt = conn.newStatement("SELECT 1")) {
      Result unusedResult = stmt.executeQuery();
    }
  }

  @Test
  public void multipleConnections_onBusy_throws_onRetry_works() throws Exception {
    try (Connection conn1 = Sqlite.newConnection(dbPath);
        Connection conn2 = Sqlite.newConnection(dbPath)) {
      conn1.executeUpdate("BEGIN");
      conn1.executeUpdate("CREATE TABLE IF NOT EXISTS tbl (id INTEGER)");
      conn2.executeUpdate("BEGIN");
      SqliteException e =
          assertThrows(
              SqliteException.class,
              () -> conn2.executeUpdate("CREATE TABLE IF NOT EXISTS tbl (id INTEGER)"));
      assertThat(e).hasMessageThat().contains("database is locked");
      assertThat(e.getCode()).isEqualTo(Sqlite.ERR_BUSY);
      conn1.executeUpdate("COMMIT");
      conn2.executeUpdate("CREATE TABLE IF NOT EXISTS tbl (id INTEGER)");
      conn2.executeUpdate("COMMIT");
    }
  }

  @Test
  public void closeConnection_withOpenStatement_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath)) {
      Statement unusedStmt = conn.newStatement("SELECT 1");
    }
  }

  @Test
  public void closeConnection_withOpenStatementAndResult_works() throws Exception {
    try (Connection conn = Sqlite.newConnection(dbPath)) {
      Statement stmt = conn.newStatement("SELECT 1");
      Result unusedResult = stmt.executeQuery();
    }
  }
}
