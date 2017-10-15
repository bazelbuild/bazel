package joptsimple.util;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Enum for checking common conditions of files and directories.
 *
 * @see joptsimple.util.PathConverter
 */
public enum PathProperties {
    FILE_EXISTING( "file.existing" ) {
        @Override
        boolean accept( Path path ) {
            return Files.isRegularFile( path );
        }
    },
    DIRECTORY_EXISTING( "directory.existing" ) {
        @Override
        boolean accept( Path path ) {
            return Files.isDirectory( path );
        }
    },
    NOT_EXISTING( "file.not.existing" ) {
        @Override
        boolean accept( Path path ) {
            return Files.notExists( path );
        }
    },
    FILE_OVERWRITABLE( "file.overwritable" ) {
        @Override
        boolean accept( Path path ) {
            return FILE_EXISTING.accept( path ) && WRITABLE.accept( path );
        }
    },
    READABLE( "file.readable" ) {
        @Override
        boolean accept( Path path ) {
            return Files.isReadable( path );
        }
    },
    WRITABLE( "file.writable" ) {
        @Override
        boolean accept( Path path ) {
            return Files.isWritable( path );
        }
    };

    private final String messageKey;

    private PathProperties( String messageKey ) {
        this.messageKey = messageKey;
    }

    abstract boolean accept( Path path );

    String getMessageKey() {
        return messageKey;
    }
}
