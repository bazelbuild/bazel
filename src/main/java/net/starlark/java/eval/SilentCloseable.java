package net.starlark.java.eval;

public interface SilentCloseable extends AutoCloseable {
    @Override
    void close();
}
