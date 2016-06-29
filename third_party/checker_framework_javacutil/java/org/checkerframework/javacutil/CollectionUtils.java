package org.checkerframework.javacutil;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Utility methods related to Java Collections
 */
public class CollectionUtils {

    /**
     * A Utility method for creating LRU cache
     * @param size  size of the cache
     * @return  a new cache with the provided size
     */
    public static <K, V> Map<K, V> createLRUCache(final int size) {
        return new LinkedHashMap<K, V>() {

            private static final long serialVersionUID = 5261489276168775084L;
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, V> entry) {
                return size() > size;
            }
        };
    }
}
