package com.google.devtools.build.lib.bazel.repository.downloader;

import java.util.Objects;
import java.util.Optional;

public interface CredentialsProvider {

    Optional<Credentials> getCredentials(String host);

    class Credentials {

        private String user = null;

        private String password = null;

        Credentials() {}

        Credentials(String user, String password) {
            this.user = user;
            this.password = password;
        }

        public Optional<String> getUser() {
            return Optional.ofNullable(user);
        }

        public Optional<String> getPassword() {
            return Optional.ofNullable(password);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Credentials that = (Credentials) o;
            return Objects.equals(user, that.user) &&
                    Objects.equals(password, that.password);
        }

        @Override
        public int hashCode() {
            return Objects.hash(user, password);
        }

        @Override
        public String toString() {
            return "Credentials{" +
                    "user='" + user + '\'' +
                    ", password='" + password + '\'' +
                    '}';
        }
    }
}
