package joptsimple.examples.ant.filters;

import java.io.IOException;
import java.io.Reader;
import java.util.HashMap;
import java.util.Map;

import org.apache.tools.ant.filters.BaseFilterReader;
import org.apache.tools.ant.filters.ChainableReader;

/**
 * Ant filter class that transforms HTML special characters into their equivalent entities.
 *
 * @author <a href="mailto:pholser@alumni.rice.edu">Paul Holser</a>
 */
public class HTMLEntitifier extends BaseFilterReader implements ChainableReader {
    private static final Map<Integer, String> ENTITIES = new HashMap<>();

    static {
        ENTITIES.put( (int) '<', "&lt;" );
        ENTITIES.put( (int) '>', "&gt;" );
        ENTITIES.put( (int) '"', "&quot;" );
        ENTITIES.put( (int) '&', "&amp;" );
    }

    private String replacementData;
    private int replacementIndex = -1;

    /**
     * Creates "dummy" instances.
     */
    public HTMLEntitifier() {
        // empty on purpose
    }

    /**
     * @param source where the data to filter comes from
     */
    public HTMLEntitifier( Reader source ) {
        super( source );
    }

    /**
     * {@inheritDoc}
     */
    public Reader chain( Reader source ) {
        HTMLEntitifier newFilter = new HTMLEntitifier( source );
        newFilter.setInitialized( true );

        return newFilter;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int read() throws IOException {
        if ( !getInitialized() )
            setInitialized( true );

        if ( replacementIndex > -1 ) {
            int ch = replacementData.charAt( replacementIndex++ );

            if ( replacementIndex >= replacementData.length() )
                replacementIndex = -1;

            return ch;
        }

        int nextChar = in.read();

        if ( ENTITIES.containsKey( nextChar ) ) {
            replacementData = ENTITIES.get( nextChar );
            replacementIndex = 1;
            return replacementData.charAt( 0 );
        }

        return nextChar;
    }
}
