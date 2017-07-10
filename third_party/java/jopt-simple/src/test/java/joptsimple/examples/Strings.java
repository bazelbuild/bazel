package joptsimple.examples;

import java.util.Iterator;

import static java.util.Arrays.*;

public class Strings {
    public static String join( char delimiter, String... pieces ) {
        StringBuilder builder = new StringBuilder();

        for ( Iterator<String> iter = asList( pieces ).iterator(); iter.hasNext(); ) {
            builder.append( iter.next() );
            if ( iter.hasNext() )
                builder.append( delimiter );
        }

        return builder.toString();
    }
}
